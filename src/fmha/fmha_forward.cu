#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include <cutlass/cutlass.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
#include "cutlass/util/cublas_wrappers.hpp"
#endif
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/util/helper_cuda.hpp"
#include "utils/cuda_launch.hpp"
#include "utils/fmha_cutlass.hpp"
#include "utils/random.hpp"

#include "gemm/copy_tensor.hpp"
#include "gemm/gemm_tensor.hpp"
#include "online_softmax.h"

// Default 64.
#ifdef QBLKSIZE
constexpr int kQueriesPerBlock = QBLKSIZE;
#else
constexpr int kQueriesPerBlock = 64;
#endif

#ifdef KBLKSIZE
constexpr int kKeysPerBlock = KBLKSIZE;
#else
constexpr int kKeysPerBlock = 64;
#endif

#ifdef HEAD
constexpr int kHeadSize = HEAD;
#else
constexpr int kHeadSize = 64;
#endif

constexpr int kT = kHeadSize;

constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
const float softmax_scale = (1.0f / sqrt(float(kHeadSize)));
const float scale = softmax_scale * kLog2e;

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

struct TStoTP {};
struct MNtoTC {};
struct MNtoTQ {};

template <typename T, typename ReshapeT> struct Reshape {

  template <class FragmentC, class FragmentQ>
  __device__ auto operator()(FragmentC &tC, FragmentQ &tQ) {
    return make_layout(1);
  }
};

// NOT TESTED or USED YET.
template <> struct Reshape<cutlass::half_t, MNtoTC> {
  template <class FragmentC, class FragmentQ>
  __device__ auto operator()(FragmentC &tMN, FragmentQ &tC) {

    static_assert(rank<0>(tC) == 3);
    static_assert(rank<1>(tC) == 1);
    static_assert(rank<2>(tC) == 1);

    auto MT = size<1>(tC);
    auto NT = size<2>(tC);
    auto VT = size<0>(tC);
    auto rowSizePerTile = 2 * size<2>(VT);
    auto rowSize = rowSizePerTile * NT;

    return make_layout(
        tC.layout().shape(),
        make_stride(make_stride(1, rowSize, 2), 2 * rowSize, rowSizePerTile));
  }
};

template <> struct Reshape<cutlass::half_t, TStoTP> {
  template <class FragmentC, class FragmentQ>
  __device__ auto operator()(FragmentC &tC, FragmentQ &tQ) {

    // get the layout of one row of Q
    auto layoutQRow = make_ordered_layout(tQ(_, 0, _).layout());
    // get the layout of  M dimension of C.
    auto layoutCM = get<1>(tC.layout());
    return make_layout(get<0>(layoutQRow), layoutCM, get<1>(layoutQRow));
  }
};

template <typename To_type, typename From_type, typename Fragment>
inline __device__ auto convert_type(Fragment const &tensor) {
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <class TiledMma0, class TiledMma1, class ClusterShape, class TA,
          class TiledCopyA, class TileShapeA, class GmemLayoutA,
          class SmemLayoutQ, class TB, class TiledCopyB, class TileShapeB,
          class GmemLayoutB, class SmemLayoutK, class TC, class TileShapeQ,
          class GmemLayoutQ, class SmemLayoutS, class TiledCopyV,
          class TileShapeV, class GmemLayoutV, class SmemLayoutV,
          class SmemLayoutVt, class TileShapeD, class GmemLayoutD,
          class GmemLayoutMI,
          class AccumType>
__global__ static void //__launch_bounds__(128, 2)
fmhaForward(TA const *Q, CUTE_GRID_CONSTANT TiledCopyA const tmaLoadQ,
            TileShapeA tileShapeQ, GmemLayoutA gmemLayoutQ,
            SmemLayoutQ smemLayoutQ, TB const *K,
            CUTE_GRID_CONSTANT TiledCopyB const tmaLoadK, TileShapeB tileShapeK,
            GmemLayoutB gmemLayoutK, SmemLayoutK smemLayoutK, TC *S,
            TileShapeQ tileShapeS, GmemLayoutQ gmemLayoutS,
            SmemLayoutS smemLayoutS, int nTilesOfK, TB *V,
            CUTE_GRID_CONSTANT TiledCopyV const tmaLoadV, TileShapeV tileShapeV,
            GmemLayoutV gmemLayoutV, SmemLayoutV smemLayoutV,
            SmemLayoutVt smemLayoutVt, TC *O, TileShapeD tileShapeO,
            GmemLayoutD gmemLayoutO, AccumType *mi_ptr, AccumType *sPrimePtr,
            GmemLayoutMI gmemLayoutMi, float scale) {

  using namespace cute;
  using X = Underscore;
  CUTE_STATIC_ASSERT_V(product_each(shape(tileShapeQ)) ==
                       product_each(shape(smemLayoutQ)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM
  // addresses
  extern __shared__ char shared_memory[];
  using SharedStorage =
      SharedStorage<TA, SmemLayoutQ, SmemLayoutK, SmemLayoutS, SmemLayoutV>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  Tensor miGlobal = make_tensor(make_gmem_ptr(mi_ptr), gmemLayoutMi);
  Tensor miGlobalOut =
      local_tile(miGlobal, make_shape(get<0>(tileShapeQ), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  Tensor sPrimeGlobal = make_tensor(make_gmem_ptr(sPrimePtr), gmemLayoutMi);
  Tensor sPrimeGlobalOut =
      local_tile(sPrimeGlobal, make_shape(get<0>(tileShapeQ), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));

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
  Tensor sVt =
      make_tensor(make_smem_ptr(shared_storage.smem_v.data()), smemLayoutVt);

  Tensor mQ = tmaLoadQ.get_tma_tensor(shape(gmemLayoutQ));
  Tensor mK = tmaLoadK.get_tma_tensor(shape(gmemLayoutK));
  Tensor mV = tmaLoadV.get_tma_tensor(shape(gmemLayoutV));
  Tensor mS = make_tensor(make_gmem_ptr(S), gmemLayoutS);
  Tensor mO = make_tensor(make_gmem_ptr(O), gmemLayoutO);
  auto blkCoordQ = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);

  Tensor gQ = local_tile(mQ, tileShapeQ, blkCoordQ);
  //
  // Prepare the TMA_LOAD for A
  //

  auto cta_tmaQ = tmaLoadQ.get_slice(0); // CTA slice
  auto cta_tmak = tmaLoadK.get_slice(0); // CTA slice
  auto cta_tmaV = tmaLoadV.get_slice(0); // CTA slice

  Tensor tQgQX = cta_tmaQ.partition_S(gQ);

  TiledMma0 tiledMma0;
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  TiledMma1 tiledMma1;
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);
  //
  // Perform the TMA_LOAD
  //

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through
  // the tiles
  Tensor tQgQ = group_modes<1, rank(tQgQX)>(tQgQX); // (TMA,REST)
  auto kTiles = size<1>(tQgQ);
  assert(kTiles == 1);
  assert(kTiles == size<2>(gQ));

  Tensor tQsQX = cta_tmaQ.partition_D(sQ);          // (TMA,TMA_M,TMA_N)
  Tensor tQsQ = group_modes<1, rank(tQsQX)>(tQsQX); // (TMA,REST)
  Tensor tKsKX = cta_tmak.partition_D(sK);
  Tensor tKsK = group_modes<1, rank(tKsKX)>(tKsKX);
  Tensor tVsVX = cta_tmaV.partition_D(sV);
  Tensor tVsV = group_modes<1, rank(tVsVX)>(tVsVX);
  static_assert(size<1>(tQsQ) == 1);
  static_assert(size<1>(tKsK) == 1);

  // Allocate "fragments/descriptors"
  // for first matmul.
  Tensor tSsQ = threadMma0.partition_A(sQ);
  Tensor tSsK = threadMma0.partition_B(sK);
  Tensor tSrQ = threadMma0.make_fragment_A(tSsQ);
  Tensor tSrK = threadMma0.make_fragment_B(tSsK);
  Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);
  clear(tSrS);
#ifdef SINSMEM
  Tensor tSsS = threadMma0.partition_C(sS);
  __syncthreads();
  cute::fill(tSsS, TC(0.0));
  __syncthreads();
#endif

  // For second mamtul, S becomes P.
  Tensor tOsP = threadMma1.partition_A(sS);
  Tensor tOrV = threadMma1.partition_fragment_B(sVt);
  Tensor tOrO = partition_fragment_C(tiledMma1, tileShapeO);
  clear(tOrO);

#ifdef SINSMEM
  Tensor tOrP = threadMma1.make_fragment_A(tOsP);
#else
  Tensor tOrS = threadMma1.make_fragment_A(tOsP);
  auto tOrPLayout = Reshape<TC, TStoTP>()(tSrS, tOrS);
  auto tOrP = make_tensor(reinterpret_cast<TA *>(tSrS.data()), tOrPLayout);
#endif

  Tensor rowMax = make_tensor<AccumType>(Shape<Int<2 * size<1>(tSrS)>>{});
  Tensor rowSum = make_fragment_like(rowMax);
  cute::fill(rowMax, -cutlass::platform::numeric_limits<AccumType>::infinity());
  cute::fill(rowSum, AccumType(0.0));

  cfk::copy(tQgQ(_, 0), tQsQ(_, 0), tmaLoadQ, tma_load_mbar[0]);

  auto blkCoordK = make_coord(0, 0, blockIdxH, blockIdxB);

  Tensor gK = local_tile(mK, tileShapeK, blkCoordK);

  Tensor tKgKX = cta_tmak.partition_S(gK); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tKgK = group_modes<1, rank(tKgKX)>(tKgKX); // (TMA,REST)
  assert(size<1>(tKgK) == size<2>(gK));
  assert(size<1>(tKgK) == kTiles);
  static_assert(size<1>(tKsK) == 1);
  cfk::copy_nobar(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0]);
#pragma unroll
  for (uint64_t blockIdxY = 0; blockIdxY < nTilesOfK; ++blockIdxY) {

    auto blkCoordV = make_coord(blockIdxY, 0, blockIdxH, blockIdxB);
    Tensor gV = local_tile(mV, tileShapeV, blkCoordV);

    Tensor tVgVX = cta_tmaV.partition_S(gV);

    Tensor tVgV = group_modes<1, rank(tVgVX)>(tVgVX);
    assert(size<1>(tVgV) == size<2>(gV));
    assert(size<1>(tVgV) == 1);

    cfk::copy_nobar(tVgV(_, 0), tVsV(_, 0), tmaLoadV, tma_load_mbar[1]);
    clear(tSrS);
    cfk::gemm_bar_wait(tiledMma0, tSrQ, tSrK, tSrS, tma_load_mbar[0]);

#ifdef COPYOUTMM0
    auto blkCoordS = make_coord(blockIdxX, blockIdxY, blockIdxH, blockIdxB);
    Tensor gS = local_tile(mS, tileShapeS, blkCoordS);
    Tensor tSgS = threadMma0.partition_C(gS);
    copy(tSrS, tSgS);
#endif

    if (blockIdxY != (nTilesOfK - 1)) {
      auto blkCoordK = make_coord(blockIdxY + 1, 0, blockIdxH, blockIdxB);

      auto gK = local_tile(mK, tileShapeK, blkCoordK);

      Tensor tKgKX = cta_tmak.partition_S(gK);
      Tensor tKgK = group_modes<1, rank(tKgKX)>(tKgKX);
      cfk::copy_nobar(tKgK(_, 0), tKsK(_, 0), tmaLoadK, tma_load_mbar[0]);
    }

#ifndef NOSMAX
    if (blockIdxY == 0) {
      onlineSoftmaxAndRescale<true, AccumType>(rowMax, rowSum, tSrS, tOrO,
                                               scale);
    } else {
      onlineSoftmaxAndRescale<false, AccumType>(rowMax, rowSum, tSrS, tOrO,
                                                scale);
    }
    warpgroup_fence_operand(tSrS);
#endif

#ifndef NOGEMM
#ifdef SINSMEM
    cfk::copy(tSrS, tSsS);
    cfk::gemm_bar_wait(tiledMma1, tOrP, tOrV, tOrO, tma_load_mbar[1]);
#else
    cfk::gemm_bar_wait(tiledMma1, convert_type<TA, AccumType>(tOrP), tOrV, tOrO,
                       tma_load_mbar[1]);
#endif
#endif
  }

#ifdef COPYOUTMI
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
  auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gO = local_tile(mO, tileShapeO, blkCoordO);
  Tensor tOgO = threadMma1.partition_C(gO);
#ifndef NOSMAX
  applySoftmaxNormalizer<AccumType>(rowSum, tOrO);
#endif

  copy(tOrO, tOgO);

  __syncthreads();
}

template <typename TA, typename TB, typename TC, typename Alpha, typename Beta,
          typename AccumType>
void fmhaForwardDevice(int SEQLEN, int KEYLEN, int HEADDIM, int NUMHEADS,
                       int BATCH, Alpha alpha, TA const *tensorQ,
                       TB const *tensorK, Beta beta, TC *tensorV, TC *tensorS,
                       TC *tensorO, AccumType *miOut, AccumType *sPrimeOut,
                       int iterations, cudaStream_t stream = 0) {
  using namespace cute;

  // Define shapes (dynamic)
  auto B = int(BATCH);
  auto H = int(NUMHEADS);
  auto M = int(SEQLEN);
  auto N = int(KEYLEN);
  auto K = int(HEADDIM);

  using ClusterShape = Shape<_1, _1, _1>;
  // Define TileShapes
  using bM = Int<kQueriesPerBlock>;
  using bN = Int<kKeysPerBlock>;
  using bK = Int<kT>;

  using MmaA = cute::conditional_t<cute::is_same_v<TA, float>, tfloat32_t, TA>;
  using MmaB = cute::conditional_t<cute::is_same_v<TB, float>, tfloat32_t, TB>;
  using MmaC = AccumType;

  // k-major for Q.
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

  // k-major for K.
  auto tileShapeK = make_shape(bN{}, bK{});
  auto smemLayoutK =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaB>{}, tileShapeK);
  Layout gmemLayoutK =
      make_layout(make_shape(N, K, H, B), make_stride(K * H, 1, K, H * N * K));
  Tensor gK = make_tensor(ptrK, gmemLayoutK);
  auto tmak =
      make_tma_copy(SM90_TMA_LOAD{}, gK, smemLayoutK, tileShapeK, Int<1>{});

  // k-major for S, where K = N.
  auto tileShapeS = make_shape(bM{}, bN{});
  // Use only durign debugging, direct writes to GMEM.
  Layout gmemLayoutS =
      make_layout(make_shape(M, N, H, B), make_stride(N, 1, N * M, H * M * N));
  // Used only for Second matmul with Q and V.
  auto smemLayoutS =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaA>{}, tileShapeS);

  // k-major for V. Layout for GEMM-II.

  auto tileShapeV = make_shape(bN{}, bK{});
  auto smemLayoutV =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<MmaB>{}, tileShapeV);
  Layout gmemLayoutV =
      make_layout(make_shape(N, K, H, B), make_stride(K * H, 1, K, H * K * N));
  Tensor gV = make_tensor(ptrV, gmemLayoutV);
  auto tmaV =
      make_tma_copy(SM90_TMA_LOAD{}, gV, smemLayoutV, tileShapeV, Int<1>{});

  // Layout for Vtranspose. For using in GMMA.
  auto tileShapeVt = make_shape(bK{}, bN{});
  using SmemLayoutVAtomBits =
      ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag,
                     Layout<Shape<_1024, Int<bN{}>>, Stride<_1, _1024>>>;
  using SmemLayoutVAtom =
      decltype(upcast<sizeof_bits<MmaB>::value>(SmemLayoutVAtomBits{}));
  auto smemLayoutVt = tile_to_shape(SmemLayoutVAtom{}, tileShapeVt);

  // k-major for D.
  auto tileShapeO = make_shape(bM{}, bK{});
  Layout gmemLayoutO =
      make_layout(make_shape(M, K, H, B), make_stride(K * H, 1, K, H * M * K));

  int smem_size = int(
      sizeof(SharedStorage<MmaA, decltype(smemLayoutQ), decltype(smemLayoutK),
                           decltype(smemLayoutS), decltype(smemLayoutV)>));

#ifdef CTA256
  using MmaTileShape = Layout<Shape<_2, _1, _1>>;
#else
  using MmaTileShape = Layout<Shape<_1, _1, _1>>;
#endif

  using TiledMma0 = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<MmaA, MmaB, MmaC, Shape<bM, bN, bK>>(),
      MmaTileShape{}));

  // For fp32 types, map to tf32 MMA value type
#ifdef SINSMEM
  using TiledMma1 = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<MmaA, MmaB, MmaC, Shape<bM, bK, bN>,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      MmaTileShape{}));
#else
  using TiledMma1 = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<MmaA, MmaB, MmaC, Shape<bM, bK, bN>,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      MmaTileShape{}));
#endif

  // col-major for MI and S_prime (only for debugging).
  Layout gmemLayoutMi = make_layout(make_shape(M, H, B), GenColMajor{});

  void const *kernel = (void const *)fmhaForward<
      TiledMma0, TiledMma1, ClusterShape, MmaA, decltype(tmaQ),
      decltype(tileShapeQ), decltype(gmemLayoutQ), decltype(smemLayoutQ), MmaB,
      decltype(tmak), decltype(tileShapeK), decltype(gmemLayoutK),
      decltype(smemLayoutK), TC, decltype(tileShapeS), decltype(gmemLayoutS),
      decltype(smemLayoutS), decltype(tmaV), decltype(tileShapeV),
      decltype(gmemLayoutV), decltype(smemLayoutV), decltype(smemLayoutVt),
      decltype(tileShapeO), decltype(gmemLayoutO), decltype(gmemLayoutMi),
      AccumType>;
  cfk::utils::set_smem_size(smem_size, kernel);

  dim3 block_dims(size(TiledMma0{}));
  dim3 grid_dims(ceil_div(size(M), size(bM{})), H, B);
  dim3 cluster_dims(cute::size<0>(ClusterShape{}),
                    cute::size<1>(ClusterShape{}),
                    cute::size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams params{grid_dims, block_dims, cluster_dims,
                                      smem_size, stream};

  auto nTilesOfK = ceil_div(size(N), size(bN{}));

  for (int i = 0; i < iterations; ++i) {
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, kernel, ptrQ, tmaQ, tileShapeQ, gmemLayoutQ, smemLayoutQ, ptrK,
        tmak, tileShapeK, gmemLayoutK, smemLayoutK, tensorS, tileShapeS,
        gmemLayoutS, smemLayoutS, nTilesOfK, tensorV, tmaV, tileShapeV,
        gmemLayoutV, smemLayoutV, smemLayoutVt, tensorO, tileShapeO,
        gmemLayoutO, miOut, sPrimeOut, gmemLayoutMi, scale);
  }
}

template <typename TA, typename TB, typename TC, typename Alpha, typename Beta,
          typename AccumType>
void fmhaForwardDeviceLoop(int SEQLEN, int KEYLEN, int HEADDIM, int NUMHEADS,
                           int BATCHSIZE, Alpha alpha, TA const *A, TB const *B,
                           Beta beta, TC *V, TC *Q, TC *D, AccumType *miOut,
                           AccumType *sPrimeOut, int iterations, int nStreams) {

  if (nStreams == 1) {
    fmhaForwardDevice(SEQLEN, KEYLEN, HEADDIM, NUMHEADS, BATCHSIZE, alpha, A, B,
                      beta, V, Q, D, miOut, sPrimeOut, iterations);
    return;
  } else {
    auto L = BATCHSIZE / nStreams;
    for (int i = 0; i < nStreams; ++i) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      auto offsetA = i * SEQLEN * HEADDIM * L;
      auto offsetB = i * KEYLEN * HEADDIM * L;
      auto offsetQ = i * SEQLEN * KEYLEN * L;
      auto offsetV = i * KEYLEN * HEADDIM * L;
      auto offsetD = i * SEQLEN * HEADDIM * L;
      auto miOffset = i * SEQLEN * L;

      fmhaForwardDevice(SEQLEN, KEYLEN, HEADDIM, NUMHEADS, L, alpha,
                        A + offsetA, B + offsetB, beta, V + offsetV,
                        Q + offsetQ, D + offsetD, miOut + miOffset,
                        sPrimeOut + miOffset, iterations, stream);
    }
  }
}

#include <cassert>
#include <cstdio>
#include <cstdlib>

template <typename TA, typename TB, typename TC, typename TI>
void testFmhaForward(int m, int n, int k, int numHeads, int batchSize,
                     int iterations, bool refCheck, bool printValues,
                     int nStreams) {
  cudaDeviceReset();
  cute::device_init(0);

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

  thrust::device_vector<TA> devQ(mLong * kLong * lLong);
  thrust::device_vector<TB> devK(nLong * kLong * lLong);
  thrust::device_vector<TC> devS(mLong * nLong * lLong);
  thrust::device_vector<TC> devV(nLong * kLong * lLong);
  thrust::device_vector<TC> devD(mLong * kLong * lLong);

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
  cfk::initialize_const(devS.data().get(), devS.size(), TC(-1));
  cfk::initialize_const(devD.data().get(), devD.size(), TC(-1));

  using AccumType = float; // AccumType is always float.

  thrust::host_vector<TA> hostQ = devQ;
  thrust::host_vector<TB> hostK = devK;
  thrust::host_vector<TC> hostS = devS;
  thrust::host_vector<TB> hostV = devV;
  thrust::host_vector<TC> hostD = devD;

  thrust::device_vector<AccumType> devMiOut(mLong * lLong);
  thrust::device_vector<AccumType> devSprimeOut(mLong * lLong);

  TI alpha = TI(1.0);
  TI beta = TI(0.0);

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
  fmhaForwardDeviceLoop(
      m, n, k, numHeads, batchSize, alpha, devQ.data().get(), devK.data().get(),
      beta, devV.data().get(), devS.data().get(), devD.data().get(),
      devMiOut.data().get(), devSprimeOut.data().get(), 10, nStreams);
  CUTE_CHECK_LAST();

  // Timing iterations
  devS = hostS;
  devD = hostD;
  devV = hostV;
  timer.start();
  fmhaForwardDeviceLoop(
      m, n, k, numHeads, batchSize, alpha, devQ.data().get(), devK.data().get(),
      beta, devV.data().get(), devS.data().get(), devD.data().get(),
      devMiOut.data().get(), devSprimeOut.data().get(), iterations, nStreams);
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s(CUTLASS) [%6.1f]Gflop/s(FLASH2)  "
         "(%6.4f)ms\n",
         cutlass_flops / cute_time, flash2_flops / cute_time, cute_time * 1000);

  thrust::host_vector<TC> cute_result_S = devS;
  thrust::host_vector<TC> cute_result_D = devD;
  thrust::host_vector<AccumType> miHostOut = devMiOut;
  thrust::host_vector<AccumType> sPrimeHostOut = devSprimeOut;
  thrust::host_vector<AccumType> miRefHostOut(miHostOut.size());
  thrust::host_vector<AccumType> sPrimeRefHostOut(sPrimeHostOut.size());

  bool usePreScaling = true;
  bool usePow2 = false;

  if (refCheck) {
    TestAttention<TC, AccumType> testBed(numHeads, batchSize, k, m);
    testBed.initialize();
    devD = hostD;
    devS = hostS;
    testBed.compute(devQ.data().get(), devK.data().get(), devV.data().get(),
                    devS.data().get(), devD.data().get(), miRefHostOut.data(),
                    sPrimeRefHostOut.data(), usePow2, usePreScaling);
    thrust::host_vector<TC> cublas_result_S = devS;
    thrust::host_vector<TC> cublas_result_D = devD;

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
    // Our intermediate write to MI is scaled with log2e. So un-scale it before checking
    // with CUTLASS.
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
      << "  --dim-size=<int>            Size of the head dimension. \n"
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

  int seqLength, batchSize, dimSize, iterations, nStreams;
  bool refCheck, printValues;
  cmd.get_cmd_line_argument("batch-size", batchSize, 16);
  cmd.get_cmd_line_argument("dim-size", dimSize, 2048);
  cmd.get_cmd_line_argument("seq-length", seqLength, 1024);
  cmd.get_cmd_line_argument("iterations", iterations, 20);
  cmd.get_cmd_line_argument("num-cuda-streams", nStreams, 1);
  cmd.get_cmd_line_argument("reference-check", refCheck, false);
  cmd.get_cmd_line_argument("print-values", printValues, false);

  int numHeads = dimSize / kHeadSize;

  testFmhaForward<cutlass::half_t, cutlass::half_t, cutlass::half_t,
                  cutlass::half_t>(seqLength, seqLength, kHeadSize, numHeads,
                                   batchSize, iterations, refCheck, printValues,
                                   nStreams);

  return 0;
}
