/***************************************************************************************************
 * Copyright (c) 2017 - 2023 COLFAX
 * TODO: Fill copyright details.
 **************************************************************************************************/

/*! \file
    \brief FMHA Attention Example.

    This workload computes a fused multi head attention.
    Because it keeps the attention matrix in shared memory, it's both faster and
    uses less global memory.

    This is based on `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_,
    and very similar to `"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" <https://arxiv.org/abs/2205.14135>`_.

    Algorithm:
      In short, we can compute the output incrementally in blocks of size B,
      we just need to divide the final result by the sum of all coefficients in
      the softmax (which we compute incrementally) with the following pseudo-code:

      ```
      s_prime = torch.zeros([num_queries, B])
      O = torch.zeros([num_queries, head_size_v])
      for i in range(0, K.shape[0], B):
        si = exp((Q . K[i * B:(i+1) * B].t) * scale)
        sum_coefs += attn_unscaled.sum(-1)
        O  += si . V[i * B:(i+1) * B]
      O = O / s_prime
      ```

      In practice, and for numerical stability reasons,
      we also substract the maximum so far (`mi`) before doing
      the exponential. When we encounter new keys, the maximum
      used to compute O so far (`m_prime`) can differ from the
      current maximum, so we update O before accumulating with

      ```
      O       = O * exp(m_prime - mi)
      m_prime = mi
      ```

    Implementation details:
      - `mi` and `si` are stored in RMEM between the 2 back to back gemms.
      - we keep and accumulate the output directly in registers.
      - blocks are parallelized across the batch dimension (B), the number
      of heads (H), and the query sequence size (M).

*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include <cutlass/cutlass.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "utils/cuda_launch.hpp"
#include "utils/fmha_cutlass.hpp"
#include "utils/random.hpp"

#include "gemm/copy_tensor.hpp"
#include "gemm/gemm_tensor.hpp"

#include "online_softmax.h"

// Default kQueriesPerBlock is 64.
#ifdef QBLKSIZE
constexpr int kQueriesPerBlock = QBLKSIZE;
#else
constexpr int kQueriesPerBlock = 64;
#endif

// Default kKeysPerBlock is 64.
#ifdef KBLKSIZE
constexpr int kKeysPerBlock = KBLKSIZE;
#else
constexpr int kKeysPerBlock = 64;
#endif

// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK,
          class SmemLayoutS, class SmemLayoutV>
struct SharedStorage {
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
#ifdef SINSMEM
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutS>> smem_s;
#endif
  cute::uint64_t tma_load_mbar[2];
};

// Reshape Utility for converting the layout from accumulator of GEMM-I
// to Operanad A of GEMM-II.
struct ReshapeTStoTP {
  template <class FragmentC, class FragmentQ>
  __device__ auto operator()(FragmentC &tC, FragmentQ &tQ) {

    // get the layout of one row of Q.
    auto layoutQRow = make_ordered_layout(tQ(_, 0, _).layout());
    // get the layout of  M dimension of C.
    auto layoutCM = get<1>(tC.layout());
    return make_layout(get<0>(layoutQRow), layoutCM, get<1>(layoutQRow));
  }
};

// Converstion Utility to convert RMEM from one type to another.
// Used for conversion from AccumType to PrecType.
template <typename To_type, typename From_type, typename Fragment>
inline __device__ auto convert_type(Fragment const &tensor) {
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // Note: this requires tensor to be "contiguous."
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// Main FMHA Device Kernel.
// PrecType = Precision of Computation used by GEMM (half_t by default).
// AccumType = Type of Accumulator used by GEMM (float by default).
// Other types are self-explanatory.
template <class PrecType, class AccumType, class TiledMma0, class TiledMma1,
          class TiledCopyQ, class TileShapeQ, class GmemLayoutQ,
          class SmemLayoutQ, class TiledCopyK, class TileShapeK,
          class GmemLayoutK, class SmemLayoutK, class TileShapeS,
          class GmemLayoutS, class SmemLayoutS, class TiledCopyV,
          class TileShapeV, class GmemLayoutV, class SmemLayoutV,
          class SmemLayoutVt, class TileShapeD, class GmemLayoutO,
          class GmemLayoutMI>
__global__ static void //__launch_bounds__(128, 2)
fmhaForward(PrecType const *Q, CUTE_GRID_CONSTANT TiledCopyQ const tmaLoadQ,
            TileShapeQ tileShapeQ, GmemLayoutQ gmemLayoutQ,
            SmemLayoutQ smemLayoutQ, PrecType const *K,
            CUTE_GRID_CONSTANT TiledCopyK const tmaLoadK, TileShapeK tileShapeK,
            GmemLayoutK gmemLayoutK, SmemLayoutK smemLayoutK, PrecType *S,
            TileShapeS tileShapeS, GmemLayoutS gmemLayoutS,
            SmemLayoutS smemLayoutS, int nTilesOfK, PrecType *V,
            CUTE_GRID_CONSTANT TiledCopyV const tmaLoadV, TileShapeV tileShapeV,
            GmemLayoutV gmemLayoutV, SmemLayoutV smemLayoutV,
            SmemLayoutVt smemLayoutVt, PrecType *O, TileShapeD tileShapeO,
            GmemLayoutO gmemLayoutO, AccumType *mi_ptr, AccumType *sPrimePtr,
            GmemLayoutMI gmemLayoutMi, float scale) {

  using namespace cute;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<PrecType, SmemLayoutQ, SmemLayoutK,
                                      SmemLayoutS, SmemLayoutV>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;

  // Get the block co-ordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // Construct SMEM tensors.
  Tensor sQ =
      make_tensor(make_smem_ptr(shared_storage.smem_q.data()), smemLayoutQ);
  Tensor sK =
      make_tensor(make_smem_ptr(shared_storage.smem_k.data()), smemLayoutK);
#ifdef SINSMEM
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.smem_s.data()), smemLayoutS);
#else
  // Just a dummy sS (with smem_v). It's required only for shape later.
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutS);
#endif
  Tensor sV =
      make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutV);

  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt =
      make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutVt);

  // Get the full un-partitioned tensors.
  // TMA tensors are sepcial tensors.
  Tensor mQ = tmaLoadQ.get_tma_tensor(shape(gmemLayoutQ));
  Tensor mK = tmaLoadK.get_tma_tensor(shape(gmemLayoutK));
  Tensor mV = tmaLoadV.get_tma_tensor(shape(gmemLayoutV));
  Tensor mO = make_tensor(make_gmem_ptr(O), gmemLayoutO);

  TiledMma0 tiledMma0;
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  TiledMma1 tiledMma1;
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);

  //
  // Prepare the TMA_LOADS. Currently, our cluster size is one.
  // So, use only the 0th slice.
  //
  auto cta_tmaQ = tmaLoadQ.get_slice(0);
  auto cta_tmak = tmaLoadK.get_slice(0);
  auto cta_tmaV = tmaLoadV.get_slice(0);

  // Get the block of Q for this CTA using the block co-ordinates.
  auto blkCoordQ = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gQ = local_tile(mQ, tileShapeQ, blkCoordQ);

  // Partition the copying of source tiles for Q among threads.
  Tensor tQgQX = cta_tmaQ.partition_S(gQ);
  // Group the REST_X modes and the TMA_X modes to easily iterate through the
  // tiles
  Tensor tQgQ = group_modes<1, rank(tQgQX)>(tQgQX); // (TMA,REST)
  auto kTiles = size<1>(tQgQ);
  assert(kTiles == 1);
  assert(kTiles == size<2>(gQ));

  //
  // Partition the copying of dest tiles for Q, K and V among threads.
  //
  Tensor tQsQX = cta_tmaQ.partition_D(sQ);
  Tensor tQsQ = group_modes<1, rank(tQsQX)>(tQsQX); // (TMA,REST)
  Tensor tKsKX = cta_tmak.partition_D(sK);
  Tensor tKsK = group_modes<1, rank(tKsKX)>(tKsKX);
  Tensor tVsVX = cta_tmaV.partition_D(sV);
  Tensor tVsV = group_modes<1, rank(tVsVX)>(tVsVX);
  static_assert(size<1>(tQsQ) == 1);
  static_assert(size<1>(tKsK) == 1);

  // Allocate "fragments/descriptors"
  // for first matmul.
  Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
  Tensor tSrK = threadMma0.partition_fragment_B(sK);
  Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);
  clear(tSrS);

  // Allocate "fragments/descriptors"
  // for second matmul.
  // Note: S becomes P.
  Tensor tOrV = threadMma1.partition_fragment_B(sVt);
  Tensor tOrO = partition_fragment_C(tiledMma1, tileShapeO);
  clear(tOrO);

// Use this flag to store result of GEMM-I to SMEM. GEMM-II
// will also read from SMEM. By default, this flag is disabled.
#ifdef SINSMEM
  Tensor tSsS = threadMma0.partition_C(sS);
  cute::fill(tSsS, PrecType(0.0));
  Tensor tOrP = threadMma1.partition_fragment_A(sS);
#else
  Tensor tOrS = threadMma1.partition_fragment_A(sS);
  auto tOrPLayout = ReshapeTStoTP()(tSrS, tOrS);
  auto tOrP = make_tensor(tSrS.data(), tOrPLayout);
#endif

  // Allocate space for per-thread rowMax and rowSum in rmem.
  Tensor rowMax = make_tensor<AccumType>(Shape<Int<2 * size<1>(tSrS)>>{});
  Tensor rowSum = make_fragment_like(rowMax);
  cute::fill(rowMax, -cutlass::platform::numeric_limits<AccumType>::infinity());
  cute::fill(rowSum, AccumType(0.0));

  // Copy Q tile from GMEM to SMEM.
  cfk::copy(tQgQ(_, 0), tQsQ(_, 0), tmaLoadQ, tma_load_mbar[0]);

  auto blkCoordK = make_coord(0, 0, blockIdxH, blockIdxB);

  Tensor gK = local_tile(mK, tileShapeK, blkCoordK);

  Tensor tKgKX = cta_tmak.partition_S(gK);
  Tensor tKgK = group_modes<1, rank(tKgKX)>(tKgKX); // (TMA,REST)
  assert(size<1>(tKgK) == size<2>(gK));
  assert(size<1>(tKgK) == kTiles);
  static_assert(size<1>(tKsK) == 1);

  // Copy first tile of K from GMEM to SMEM.
  cfk::copy_nobar(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0]);

#pragma unroll
  for (uint64_t blockIdxY = 0; blockIdxY < nTilesOfK; ++blockIdxY) {

    auto blkCoordV = make_coord(blockIdxY, 0, blockIdxH, blockIdxB);
    Tensor gV = local_tile(mV, tileShapeV, blkCoordV);

    Tensor tVgVX = cta_tmaV.partition_S(gV);

    Tensor tVgV = group_modes<1, rank(tVgVX)>(tVgVX);
    assert(size<1>(tVgV) == size<2>(gV));
    assert(size<1>(tVgV) == 1);

    // Copy current tile of V from GMEM to SMEM.
    cfk::copy_nobar(tVgV(_, 0), tVsV(_, 0), tmaLoadV, tma_load_mbar[1]);
    clear(tSrS);

    // Issue GEMM-I.
    cfk::gemm_bar_wait(tiledMma0, tSrQ, tSrK, tSrS, tma_load_mbar[0]);

// Required for verification ONLY.
#ifdef COPYOUTMM0
    Tensor mS = make_tensor(make_gmem_ptr(S), gmemLayoutS);
    auto blkCoordS = make_coord(blockIdxX, blockIdxY, blockIdxH, blockIdxB);
    Tensor gS = local_tile(mS, tileShapeS, blkCoordS);
    Tensor tSgS = threadMma0.partition_C(gS);
    copy(tSrS, tSgS);
#endif

    // Copy next tile of K from GMEM to SMEM.
    if (blockIdxY != (nTilesOfK - 1)) {
      auto blkCoordK = make_coord(blockIdxY + 1, 0, blockIdxH, blockIdxB);

      auto gK = local_tile(mK, tileShapeK, blkCoordK);

      Tensor tKgKX = cta_tmak.partition_S(gK);
      Tensor tKgK = group_modes<1, rank(tKgKX)>(tKgKX);
      cfk::copy_nobar(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0]);
    }

    if (blockIdxY == 0) { // Compute Online Softmax and NO Output Rescaling.
      onlineSoftmaxAndRescale<true, AccumType>(rowMax, rowSum, tSrS, tOrO,
                                               scale);
    } else { // Compute Online Softmax and Output Rescaling.
      onlineSoftmaxAndRescale<false, AccumType>(rowMax, rowSum, tSrS, tOrO,
                                                scale);
    }
    warpgroup_fence_operand(tSrS);

#ifdef SINSMEM
    // ISSUE GEMM-II with Operand A from SMEM.
    // Copy OperandA from RMEM to SMEM before issuing.
    cfk::copy(tSrS, tSsS);
    cfk::gemm_bar_wait(tiledMma1, tOrP, tOrV, tOrO, tma_load_mbar[1]);
#else
    // ISSUE GEMM-II with Operand A from RMEM.
    // Convert Opreand A From AccumType [=float] to PrecType [=half_t] before
    // issuing.
    cfk::gemm_bar_wait(tiledMma1, convert_type<PrecType, AccumType>(tOrP), tOrV,
                       tOrO, tma_load_mbar[1]);
#endif
  }

  // Copy output Tile from RMEM to GMEM directly.
  auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gO = local_tile(mO, tileShapeO, blkCoordO);
  Tensor tOgO = threadMma1.partition_C(gO);

  // Apply softmax normalization before writing out to GMEM.
  applySoftmaxNormalizer<AccumType>(rowSum, tOrO);

  // Write out to GMEM.
  copy(tOrO, tOgO);

// Write out rowMax and rowSum to GMEM.
// Required for verification ONLY.
#ifdef COPYOUTMI
  Tensor miGlobal = make_tensor(make_gmem_ptr(mi_ptr), gmemLayoutMi);
  Tensor miGlobalOut =
      local_tile(miGlobal, make_shape(get<0>(tileShapeQ), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  Tensor sPrimeGlobal = make_tensor(make_gmem_ptr(sPrimePtr), gmemLayoutMi);
  Tensor sPrimeGlobalOut =
      local_tile(sPrimeGlobal, make_shape(get<0>(tileShapeQ), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  if (threadIdx.x % 4 == 0) {
    auto mmaThreadLayoutC = TiledMma0{}.get_layoutC_TV();
    auto mmaShapeMNK = cute::tile_shape(TiledMma0{});
    auto mmaShapeMN = make_shape(shape<0>(mmaShapeMNK), shape<1>(mmaShapeMNK));
    auto flatCoord =
        cute::idx2crd(mmaThreadLayoutC(threadIdx.x, 0), mmaShapeMN);
    auto rowIdGlobal = get<0>(flatCoord); // starting rowId
    auto rowId = 0;
    for (int i = rowIdGlobal; i < kQueriesPerBlock; i += 64) {
      miGlobalOut(i) = rowMax(rowId);
      sPrimeGlobalOut(i) = rowSum(rowId);
      miGlobalOut(i + 8) = rowMax(rowId + 1);
      sPrimeGlobalOut(i + 8) = rowSum(rowId + 1);
      rowId += 2;
    }
  }
#endif

  __syncthreads();
}

// Host method that prepares the data structures
// required before calling the DEVICE kernel.
template <typename PrecType, typename AccumType, int HEADDIM>
void fmhaForwardDevice(int SEQLEN, int KEYLEN, int NUMHEADS, int BATCH,
                       PrecType const *tensorQ, PrecType const *tensorK,
                       PrecType *tensorV, PrecType *tensorS, PrecType *tensorO,
                       AccumType *miOut, AccumType *sPrimeOut, int iterations,
                       float scale, cudaStream_t stream = 0) {
  using namespace cute;

  // Define shapes (dynamic)
  auto B = int(BATCH);
  auto H = int(NUMHEADS);
  auto M = int(SEQLEN);
  auto N = int(KEYLEN);
  auto K = int(HEADDIM);

  // Define TileShapes
  using bM = Int<kQueriesPerBlock>;
  using bN = Int<kKeysPerBlock>;
  using bK = Int<HEADDIM>;

  // Use half_t for computing. float for accumulator.
  using MmaA = PrecType;
  using MmaB = PrecType;
  using MmaC = AccumType;

  //
  // All the tensors are stored in BMHK order, with K being the unit-1
  // dimension. For now, n=m.
  //

  auto ptrQ = reinterpret_cast<MmaA const *>(tensorQ);
  auto ptrK = reinterpret_cast<MmaB const *>(tensorK);
  auto ptrV = reinterpret_cast<MmaB const *>(tensorV);
  auto tileShapeQ = make_shape(bM{}, bK{});
  auto smemLayoutQ =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaA>{}, tileShapeQ);
  Layout gmemLayoutQ =
      make_layout(make_shape(M, K, H, B), make_stride(K * H, 1, K, H * M * K));
  Tensor gQ = make_tensor(ptrQ, gmemLayoutQ);
  auto tmaQ =
      make_tma_copy(SM90_TMA_LOAD{}, gQ, smemLayoutQ, tileShapeQ, Int<1>{});

  auto tileShapeK = make_shape(bN{}, bK{});
  auto smemLayoutK =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaB>{}, tileShapeK);
  Layout gmemLayoutK =
      make_layout(make_shape(N, K, H, B), make_stride(K * H, 1, K, H * N * K));
  Tensor gK = make_tensor(ptrK, gmemLayoutK);
  auto tmak =
      make_tma_copy(SM90_TMA_LOAD{}, gK, smemLayoutK, tileShapeK, Int<1>{});

  // Use only durign debugging, direct writes to GMEM.
  auto tileShapeS = make_shape(bM{}, bN{});
  Layout gmemLayoutS =
      make_layout(make_shape(M, N, H, B), make_stride(N, 1, N * M, H * M * N));
  // Used only for Second matmul with Q and V.
  auto smemLayoutS =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaA>{}, tileShapeS);

  auto tileShapeV = make_shape(bN{}, bK{});
  auto smemLayoutV =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaB>{}, tileShapeV);
  Layout gmemLayoutV =
      make_layout(make_shape(N, K, H, B), make_stride(K * H, 1, K, H * K * N));
  Tensor gV = make_tensor(ptrV, gmemLayoutV);
  auto tmaV =
      make_tma_copy(SM90_TMA_LOAD{}, gV, smemLayoutV, tileShapeV, Int<1>{});

  // Layout for Vtranspose. For using in GEMM-II.
  auto tileShapeVt = make_shape(bK{}, bN{});
  using SmemLayoutVAtomBits =
      ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag,
                     Layout<Shape<_1024, Int<bN{}>>, Stride<_1, _1024>>>;
  using SmemLayoutVAtom =
      decltype(upcast<sizeof_bits<MmaB>::value>(SmemLayoutVAtomBits{}));
  auto smemLayoutVt = tile_to_shape(SmemLayoutVAtom{}, tileShapeVt);

  auto tileShapeO = make_shape(bM{}, bK{});
  Layout gmemLayoutO =
      make_layout(make_shape(M, K, H, B), make_stride(K * H, 1, K, H * M * K));

// Enable this flag for 256 threads (or 8 warps) per CTA.
// Disabled by default.
#ifdef CTA256
  using MmaTileShape = Layout<Shape<_2, _1, _1>>;
#else
  using MmaTileShape = Layout<Shape<_1, _1, _1>>;
#endif

  // USE SS version of GMMA for GEMM-I.
  // NOTE: RS version has a synchronization bug and hence not used.
  using TiledMma0 = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<MmaA, MmaB, MmaC, Shape<bM, bN, bK>>(),
      MmaTileShape{}));

#ifdef SINSMEM
  // USE SS version of GMMA for GEMM-II.
  using TiledMma1 = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<MmaA, MmaB, MmaC, Shape<bM, bK, bN>,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      MmaTileShape{}));
#else
  // USE RS version of GMMA for GEMM-II (Default).
  using TiledMma1 = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<MmaA, MmaB, MmaC, Shape<bM, bK, bN>,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      MmaTileShape{}));
#endif

  // col-major for MI and S_prime (used only for verification).
  Layout gmemLayoutMi = make_layout(make_shape(M, H, B), GenColMajor{});

  // Get the ptr to kernel function.
  void const *kernel = (void const *)fmhaForward<
      PrecType, AccumType, TiledMma0, TiledMma1, decltype(tmaQ),
      decltype(tileShapeQ), decltype(gmemLayoutQ), decltype(smemLayoutQ),
      decltype(tmak), decltype(tileShapeK), decltype(gmemLayoutK),
      decltype(smemLayoutK), decltype(tileShapeS), decltype(gmemLayoutS),
      decltype(smemLayoutS), decltype(tmaV), decltype(tileShapeV),
      decltype(gmemLayoutV), decltype(smemLayoutV), decltype(smemLayoutVt),
      decltype(tileShapeO), decltype(gmemLayoutO), decltype(gmemLayoutMi)>;

  // Compuate and set dynamic shared memory size.
  auto smem_size = int(
      sizeof(SharedStorage<MmaA, decltype(smemLayoutQ), decltype(smemLayoutK),
                           decltype(smemLayoutS), decltype(smemLayoutV)>));
  cfk::utils::set_smem_size(smem_size, kernel);

  //
  // Define CUDA launch kernel parameters.
  //

  // Set the THREAD BLOCK (CTA) dimensions.
  // #threads in CTA = #threads in MMA (128 by default).
  dim3 block_dims(size(TiledMma0{}));

  // Set the GRID dimensions (3-D).
  // First dimension = # of blocks of Q.
  // Second dimension = # of heads.
  // Third dimension = # of batches.
  dim3 grid_dims(ceil_div(size(M), size(bM{})), H, B);

  // We do not use cluster feature yet. So, set it to 1.
  dim3 cluster_dims(1, 1, 1);

  // Define the cluster launch parameter structure.
  cutlass::ClusterLaunchParams params{grid_dims, block_dims, cluster_dims,
                                      smem_size, stream};

  // Compute the no. of tiles of K matrix.
  auto nTilesOfK = ceil_div(size(N), size(bN{}));

  // Run the CUDA kernel for preferred number of iteations.
  for (int i = 0; i < iterations; ++i) {
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel, ptrQ, tmaQ, tileShapeQ, gmemLayoutQ, smemLayoutQ, ptrK,
        tmak, tileShapeK, gmemLayoutK, smemLayoutK, tensorS, tileShapeS,
        gmemLayoutS, smemLayoutS, nTilesOfK, tensorV, tmaV, tileShapeV,
        gmemLayoutV, smemLayoutV, smemLayoutVt, tensorO, tileShapeO,
        gmemLayoutO, miOut, sPrimeOut, gmemLayoutMi, scale);
  }
}

// Wrapper function for mulitple-streams.
// Currenlty, only single stream is used by default.
template <typename PrecType, typename AccumType, int HEADDIM>
void fmhaForwardDeviceLoop(int SEQLEN, int KEYLEN, int NUMHEADS, int BATCHSIZE,
                           PrecType const *A, PrecType const *B, PrecType *V,
                           PrecType *Q, PrecType *D, AccumType *miOut,
                           AccumType *sPrimeOut, int iterations, int nStreams,
                           float scale) {

  if (nStreams == 1) {
    fmhaForwardDevice<PrecType, AccumType, HEADDIM>(
        SEQLEN, KEYLEN, NUMHEADS, BATCHSIZE, A, B, V, Q, D, miOut, sPrimeOut,
        iterations, scale);
    return;
  } else {
    auto L = BATCHSIZE / nStreams;
    for (int i = 0; i < nStreams; ++i) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      auto offsetA = i * SEQLEN * NUMHEADS * HEADDIM * L;
      auto offsetB = i * KEYLEN * NUMHEADS * HEADDIM * L;
      auto offsetQ = i * SEQLEN * NUMHEADS * KEYLEN * L;
      auto offsetV = i * KEYLEN * NUMHEADS * HEADDIM * L;
      auto offsetD = i * SEQLEN * NUMHEADS * HEADDIM * L;
      auto miOffset = i * SEQLEN * NUMHEADS * L;

      fmhaForwardDevice<PrecType, AccumType, HEADDIM>(
          SEQLEN, KEYLEN, NUMHEADS, L, A + offsetA, B + offsetB, V + offsetV,
          Q + offsetQ, D + offsetD, miOut + miOffset, sPrimeOut + miOffset,
          iterations, scale, stream);
    }
  }
}

#include <cassert>
#include <cstdio>
#include <cstdlib>

//  The main driver function.
template <typename PrecType, int HEADDIM>
void testFmhaForward(int m, int n, int numHeads, int batchSize, int iterations,
                     bool refCheck, bool printValues, int nStreams) {
  constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
  const float softmax_scale = (1.0f / sqrt(float(HEADDIM)));
  const float scale = softmax_scale * kLog2e;
  cudaDeviceReset();
  cute::device_init(0);
  int k = HEADDIM;
  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "QBLK = " << kQueriesPerBlock << std::endl;
  std::cout << "KBLK = " << kKeysPerBlock << std::endl;

  auto mLong = uint64_t(m);
  auto nLong = uint64_t(n);
  auto kLong = uint64_t(k);
  auto lLong = uint64_t(numHeads * batchSize);
  std::cout << "L = " << lLong << " : " << numHeads << " * " << batchSize
            << std::endl;

  thrust::device_vector<PrecType> devQ(mLong * kLong * lLong);
  thrust::device_vector<PrecType> devK(nLong * kLong * lLong);
  thrust::device_vector<PrecType> devS(mLong * nLong * lLong);
  thrust::device_vector<PrecType> devV(nLong * kLong * lLong);
  thrust::device_vector<PrecType> devD(mLong * kLong * lLong);

  uint32_t seed = 3080;

  cutlass::Distribution::Kind initQ;
  cutlass::Distribution::Kind initK;
  cutlass::Distribution::Kind initV;
  if (refCheck) {
    initQ = cutlass::Distribution::Uniform;
    initK = cutlass::Distribution::Uniform;
    initV = cutlass::Distribution::Uniform;
  } else {
    initQ = cutlass::Distribution::Gaussian;
    initK = cutlass::Distribution::Gaussian;
    initV = cutlass::Distribution::Gaussian;
  }

  cfk::initialize_rand(devQ.data().get(), devQ.size(), initQ, seed + 1);
  cfk::initialize_rand(devK.data().get(), devK.size(), initK, seed + 2);
  cfk::initialize_rand(devV.data().get(), devV.size(), initV, seed + 3);
  cfk::initialize_const(devS.data().get(), devS.size(), PrecType(-1));
  cfk::initialize_const(devD.data().get(), devD.size(), PrecType(-1));

  using AccumType = float; // AccumType is always float.

  thrust::host_vector<PrecType> hostQ = devQ;
  thrust::host_vector<PrecType> hostK = devK;
  thrust::host_vector<PrecType> hostS = devS;
  thrust::host_vector<PrecType> hostV = devV;
  thrust::host_vector<PrecType> hostD = devD;

  thrust::device_vector<AccumType> devMiOut(mLong * lLong);
  thrust::device_vector<AccumType> devSprimeOut(mLong * lLong);

  const int timing_iterations = iterations;
  GPU_Clock timer;

  //
  // CuTe
  //

  double cutlass_flops = 0.0;
  // P <- Q . K_t
  cutlass_flops += 2 * mLong * nLong * kLong;
  // P <- exp(P - max(P))
  cutlass_flops += 2 * mLong * nLong;
  // S <- sum(P)
  cutlass_flops += mLong * (nLong - 1);
  // O <- P . V
  cutlass_flops += 2 * mLong * nLong * kLong;
  // O <- O / S
  cutlass_flops += mLong * nLong * kLong;

  cutlass_flops *= numHeads * batchSize / double(1.0e9);

  double flash2_flops =
      4 * batchSize * numHeads * mLong * nLong * kLong / double(1.0e9);

  // Run few times (warmup).
  devS = hostS;
  devD = hostD;
  devV = hostV;
  fmhaForwardDeviceLoop<PrecType, AccumType, HEADDIM>(
      m, n, numHeads, batchSize, devQ.data().get(), devK.data().get(),
      devV.data().get(), devS.data().get(), devD.data().get(),
      devMiOut.data().get(), devSprimeOut.data().get(), 10, nStreams, scale);
  CUTE_CHECK_LAST();

  // Timing iterations
  devS = hostS;
  devD = hostD;
  devV = hostV;
  timer.start();
  fmhaForwardDeviceLoop<PrecType, AccumType, HEADDIM>(
      m, n, numHeads, batchSize, devQ.data().get(), devK.data().get(),
      devV.data().get(), devS.data().get(), devD.data().get(),
      devMiOut.data().get(), devSprimeOut.data().get(), iterations, nStreams,
      scale);
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_FMHA:     [%6.1f]GFlop/s(CUTLASS) [%6.1f]Gflop/s(FLASH2)  "
         "(%6.4f)ms\n",
         cutlass_flops / cute_time, flash2_flops / cute_time, cute_time * 1000);

  thrust::host_vector<PrecType> cute_result_S = devS;
  thrust::host_vector<PrecType> cute_result_D = devD;
  thrust::host_vector<AccumType> miHostOut = devMiOut;
  thrust::host_vector<AccumType> sPrimeHostOut = devSprimeOut;
  thrust::host_vector<AccumType> miRefHostOut(miHostOut.size());
  thrust::host_vector<AccumType> sPrimeRefHostOut(sPrimeHostOut.size());

  bool usePreScaling = true;
  bool usePow2 = false;

  if (refCheck) {
    TestAttention<PrecType, AccumType> testBed(numHeads, batchSize, k, m);
    testBed.initialize();
    devD = hostD;
    devS = hostS;
    testBed.compute(devQ.data().get(), devK.data().get(), devV.data().get(),
                    devS.data().get(), devD.data().get(), miRefHostOut.data(),
                    sPrimeRefHostOut.data(), usePow2, usePreScaling);
    thrust::host_vector<PrecType> cublas_result_S = devS;
    thrust::host_vector<PrecType> cublas_result_D = devD;

#ifdef COPYOUTMM0
    // Our intermediate write to S is unscaled. So scale it before checking with
    // CUTLASS.
    if (usePreScaling) {
      for (int j = 0; j < cute_result_S.size(); ++j) {
        cute_result_S[j] = cute_result_S[j] * softmax_scale;
      }
    }
    bool gemm1 =
        cfk::verify_tensor(cute_result_S, cublas_result_S, printValues);
    std::string result1 = gemm1 ? "Passed" : "Failed";
    std::cout << "gemm-check-1: " << result1 << std::endl;
#endif

#ifdef COPYOUTMI
    // Our intermediate write to MI is scaled with log2e. So un-scale it before
    // checking with CUTLASS.
    if (!usePow2) {
      for (int j = 0; j < miHostOut.size(); ++j) {
        miHostOut[j] = miHostOut[j] * (1.0 / kLog2e);
      }
    }
    bool maxCheck = cfk::verify_tensor(miHostOut, miRefHostOut, printValues);
    std::string maxCheckResult = maxCheck ? "Passed" : "Failed";
    std::cout << "max-check: " << maxCheckResult << std::endl;

    // Our intermediate write to sPrime is not reciprocal. So invert it before
    // checking with CUTLASS.
    for (int j = 0; j < sPrimeHostOut.size(); ++j) {
      sPrimeHostOut[j] = (1.0 / sPrimeHostOut[j]);
    }
    bool sumCheck =
        cfk::verify_tensor(sPrimeHostOut, sPrimeRefHostOut, printValues);
    std::string sumCheckResult = sumCheck ? "Passed" : "Failed";
    std::cout << "sum-check: " << sumCheckResult << std::endl;
#endif

    bool gemm2 =
        cfk::verify_tensor(cute_result_D, cublas_result_D, printValues);
    std::string result2 = gemm2 ? "Passed" : "Failed";
    std::cout << "gemm-check-2: " << result2 << std::endl;
  }
}

/// Prints the usage statement.
void print_usage() {

  std::cout
      << "fmha_forward "
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage "
         "statement.\n\n"
      << "  --batch-size=<int>          Batch size in multi-head attention "
         "(default: --batch_size=16).\n"
      << "  --dim-size=<int>            Full Size of the head dimension "
         "(before reshape). \n"
      << "  --head-size=<int>           Size of the head dimension (after "
         "reshape). \n"
      << "  --seq-length=<int>          Sequence length in multi-head "
         "attention for Q (default: --seq_length=1024).\n"
      << "  --iterations=<int>          Number of profiling iterations to "
         "perform.\n"
      << "  --num-cuda-streams=<int>    Number of CUDA streams to use "
         "(default=1).\n"
      << "  --reference-check=<bool>    If true, performs reference check.\n"
      << "  --print-values=<bool>       If true, prints the values of the "
         "result (also the intermediate results of gemm-I & softmax).\n";
}

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line
  if (cmd.check_cmd_line_flag("help")) {
    print_usage();
    return;
  }

  int seqLength, batchSize, dimSize, iterations, nStreams, kHeadSize;
  bool refCheck, printValues;
  cmd.get_cmd_line_argument("batch-size", batchSize, 16);
  cmd.get_cmd_line_argument("dim-size", dimSize, 2048);
  cmd.get_cmd_line_argument("head-size", kHeadSize, 64);
  cmd.get_cmd_line_argument("seq-length", seqLength, 1024);
  cmd.get_cmd_line_argument("iterations", iterations, 20);
  cmd.get_cmd_line_argument("num-cuda-streams", nStreams, 1);
  cmd.get_cmd_line_argument("reference-check", refCheck, false);
  cmd.get_cmd_line_argument("print-values", printValues, false);

  if (nStreams > batchSize) {
    std::cout << "#max no. of cuda streams <= batchSize" << std::endl;
    exit(-1);
  }
  int numHeads = dimSize / kHeadSize;

  // Instantiate the function template for different HEADDIMS.
  // For now, only half_t is supported. TF32 is WIP.
  if (kHeadSize == 64) {
    testFmhaForward<cutlass::half_t, 64>(seqLength, seqLength, numHeads,
                                         batchSize, iterations, refCheck,
                                         printValues, nStreams);
  } else if (kHeadSize == 128) {
    testFmhaForward<cutlass::half_t, 128>(seqLength, seqLength, numHeads,
                                          batchSize, iterations, refCheck,
                                          printValues, nStreams);
  } else if (kHeadSize == 256) {
    testFmhaForward<cutlass::half_t, 256>(seqLength, seqLength, numHeads,
                                          batchSize, iterations, refCheck,
                                          printValues, nStreams);
  } else {
    std::cout << "Unsupported head dim: " << kHeadSize << std::endl;
    exit(-1);
  }

  return 0;
}
