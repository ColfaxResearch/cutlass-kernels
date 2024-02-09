#pragma once

#include "online_softmax.h"
#include "reg2reg.h"
#include "shared_storage.h"

// Conversion Utility to convert RMEM from one type to another.
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

// FMHA Consumer does GEMMs and softmax
template <class Gemm1Type, class AccumType, class SoftType, class Gemm2Type,
          class TiledMma0, class TiledMma1, class TileShapeS, class GmemLayoutS,
          typename TensorQ, typename TensorK, typename TensorS,
          typename TensorV, typename TensorO, typename RegLayout,
          typename RowMax, typename RowSum>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardConsumer(Gemm1Type const *Q, Gemm1Type const *K, Gemm2Type const *V,
                    Gemm1Type *S, const TensorQ &tSrQ, const TensorK &tSrK,
                    TensorS &&tSrS, const TensorV &tOrV, TensorO &tOrO,
                    const RegLayout &tOrPLayout, RowMax &rowMax, RowSum &rowSum,
                    const TileShapeS &tileShapeS,
                    const GmemLayoutS &gmemLayoutS, float scale, int blockIdxY,
                    const TiledMma0 &tiledMma0, const TiledMma1 &tiledMma1,
                    const AccumType &, const SoftType &) {

  using namespace cute;

  clear(tSrS);

  // Issue GEMM-I.
  cfk::gemm(tiledMma0, tSrQ, tSrK, tSrS);

// Required for verification ONLY.
#ifdef COPYOUTMM0
  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  Tensor mS = make_tensor(make_gmem_ptr(S), gmemLayoutS);
  auto blkCoordS = make_coord(blockIdxX, blockIdxY, blockIdxH, blockIdxB);
  Tensor gS = local_tile(mS, tileShapeS, blkCoordS);
  Tensor tSgS = threadMma0.partition_C(gS);
  copy(tSrS, tSgS);
#endif

#if 1
  if (blockIdxY == 0) { // Compute Online Softmax and NO Output Rescaling.
    onlineSoftmaxAndRescale<true, SoftType>(rowMax, rowSum, tSrS, tOrO, scale);
  } else { // Compute Online Softmax and Output Rescaling.
    onlineSoftmaxAndRescale<false, SoftType>(rowMax, rowSum, tSrS, tOrO, scale);
  }
  warpgroup_fence_operand(tSrS);
#endif

#if 1
  // ISSUE GEMM-II with Operand A from RMEM.
  // Convert Operand A From SoftType [=float or half] to Gemm1Type [=half_t or
  // fp8] before issuing.
  auto tSrSPrec = convert_type<Gemm2Type, AccumType>(tSrS);
  reorgCtoA<Gemm2Type, Gemm2Type>(tSrSPrec);
  auto tOrP = make_tensor(tSrSPrec.data(), tOrPLayout);
  warpgroup_fence_operand(tSrS);
  cfk::gemm(tiledMma1, tOrP, tOrV, tOrO);
#endif
}

// Epilogue that copies RMEM -> GMEM directly
// Reports as uncoalesced stores by the profiler
template <class TensorO, class RowMax, class RowSum, class SoftType,
          class OutputType, class TiledMma0, class TiledMma1, class TileShapeO,
          class GmemLayoutO, class GmemLayoutMI>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardWriteOut(TensorO &tOrO, const RowMax &rowMax, const RowSum &rowSum,
                    OutputType *O, TileShapeO tileShapeO,
                    GmemLayoutO gmemLayoutO, SoftType *mi_ptr,
                    SoftType *sPrimePtr, GmemLayoutMI gmemLayoutMi,
                    const TiledMma0 &tiledMma0, const TiledMma1 &tiledMma1) {

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // Apply softmax normalization before writing out to GMEM.
  applySoftmaxNormalizer<SoftType>(rowSum, tOrO);

  Tensor mO = make_tensor(make_gmem_ptr(O), gmemLayoutO); 
  auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gO = local_tile(mO, tileShapeO, blkCoordO);
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);
  Tensor tOgO = threadMma1.partition_C(gO);

  // Write out to GMEM.
  copy(tOrO, tOgO);

// Write out rowMax and rowSum to GMEM.
// Required for verification ONLY.
#ifdef COPYOUTMI
  Tensor miGlobal = make_tensor(make_gmem_ptr(mi_ptr), gmemLayoutMi);
  Tensor miGlobalOut =
      local_tile(miGlobal, make_shape(get<0>(tileShapeO), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  Tensor sPrimeGlobal = make_tensor(make_gmem_ptr(sPrimePtr), gmemLayoutMi);
  Tensor sPrimeGlobalOut =
      local_tile(sPrimeGlobal, make_shape(get<0>(tileShapeO), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  if (threadIdx.x % 4 == 0) {
    auto mmaThreadLayoutC = TiledMma0{}.get_layoutC_TV();
    auto mmaShapeMNK = cute::tile_shape(TiledMma0{});
    auto mmaShapeMN = make_shape(shape<0>(mmaShapeMNK), shape<1>(mmaShapeMNK));
    auto flatCoord =
        cute::idx2crd(mmaThreadLayoutC(threadIdx.x, 0), mmaShapeMN);
    auto rowIdGlobal = get<0>(flatCoord); // starting rowId
    auto rowId = 0;
    for (int i = rowIdGlobal; i < kQueriesPerBlock;
         i += NumMmaWarpGroups * 64) {
      miGlobalOut(i) = rowMax(rowId);
      sPrimeGlobalOut(i) = rowSum(rowId);
      miGlobalOut(i + 8) = rowMax(rowId + 1);
      sPrimeGlobalOut(i + 8) = rowSum(rowId + 1);
      rowId += 2;
    }
  }
#endif
}

// Epilogue with SMEM pass-through
template <class TensorO, class RowMax, class RowSum, class SoftType,
          class OutputType, class TiledMma0, class TiledMma1, class TileShapeO,
          class GmemLayoutO, class GmemLayoutMI, class TensorSO,
          class ThreadLayoutO>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardWriteOutCoalesced(
    TensorO &tOrO, const RowMax &rowMax, const RowSum &rowSum, OutputType *O,
    TileShapeO tileShapeO, GmemLayoutO gmemLayoutO, SoftType *mi_ptr,
    SoftType *sPrimePtr, GmemLayoutMI gmemLayoutMi, const TiledMma0 &tiledMma0,
    const TiledMma1 &tiledMma1, const TensorSO &sO, ThreadLayoutO tO) {

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // Apply softmax normalization before writing out to GMEM.
  applySoftmaxNormalizer<SoftType>(rowSum, tOrO);

  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);
  Tensor tOsOAcc = threadMma1.partition_C(sO);
  cfk::copy(tOrO, tOsOAcc);

  Tensor mO = make_tensor(make_gmem_ptr(O), gmemLayoutO);  
  auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gO = local_tile(mO, tileShapeO, blkCoordO);

  auto tOgO =
      local_partition(gO, tO, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)
  auto tOsO =
      local_partition(sO, tO, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)
  
  // Write out to GMEM.
  copy(tOsO, tOgO);

// Write out rowMax and rowSum to GMEM.
// Required for verification ONLY.
#ifdef COPYOUTMI
  Tensor miGlobal = make_tensor(make_gmem_ptr(mi_ptr), gmemLayoutMi);
  Tensor miGlobalOut =
      local_tile(miGlobal, make_shape(get<0>(tileShapeO), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  Tensor sPrimeGlobal = make_tensor(make_gmem_ptr(sPrimePtr), gmemLayoutMi);
  Tensor sPrimeGlobalOut =
      local_tile(sPrimeGlobal, make_shape(get<0>(tileShapeO), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  if (threadIdx.x % 4 == 0) {
    auto mmaThreadLayoutC = TiledMma0{}.get_layoutC_TV();
    auto mmaShapeMNK = cute::tile_shape(TiledMma0{});
    auto mmaShapeMN = make_shape(shape<0>(mmaShapeMNK), shape<1>(mmaShapeMNK));
    auto flatCoord =
        cute::idx2crd(mmaThreadLayoutC(threadIdx.x, 0), mmaShapeMN);
    auto rowIdGlobal = get<0>(flatCoord); // starting rowId
    auto rowId = 0;
    for (int i = rowIdGlobal; i < kQueriesPerBlock;
         i += NumMmaWarpGroups * 64) {
      miGlobalOut(i) = rowMax(rowId);
      sPrimeGlobalOut(i) = rowSum(rowId);
      miGlobalOut(i + 8) = rowMax(rowId + 1);
      sPrimeGlobalOut(i + 8) = rowSum(rowId + 1);
      rowId += 2;
    }
  }
#endif
}

// Epilogue with TMA Store
template <class TensorO, class RowMax, class RowSum, class SoftType,
          class OutputType, class TiledMma0, class TiledMma1, class TileShapeO,
          class GmemLayoutO, class GmemLayoutMI, class TensorSO,
          class TiledCopyO>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardWriteOutTMA(TensorO &tOrO, const RowMax &rowMax,
                       const RowSum &rowSum, OutputType *O,
                       TileShapeO tileShapeO, GmemLayoutO gmemLayoutO,
                       SoftType *mi_ptr, SoftType *sPrimePtr,
                       GmemLayoutMI gmemLayoutMi, const TiledMma0 &tiledMma0,
                       const TiledMma1 &tiledMma1, const TensorSO &sO,
                       TiledCopyO const &tmaStoreO, bool leaderWarp) {

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);

  // Apply softmax normalization before writing out to GMEM.
  applySoftmaxNormalizer<SoftType>(rowSum, tOrO);

  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);
  Tensor tOsOAcc = threadMma1.partition_C(sO);
  cfk::copy(tOrO, tOsOAcc);
  
  Tensor mO = tmaStoreO.get_tma_tensor(shape(gmemLayoutO));
  auto blkCoordO = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gO = local_tile(mO, tileShapeO, blkCoordO);
  
  auto cta_tmaO = tmaStoreO.get_slice(0);
  
  Tensor tOgOX = cta_tmaO.partition_D(gO);  
  Tensor tOgO = group_modes<1, rank(tOgOX)>(tOgOX); // (TMA,REST)
  assert(size<1>(tOgO) == 1);

  Tensor tOsOX = cta_tmaO.partition_S(sO);
  Tensor tOsO = group_modes<1, rank(tOsOX)>(tOsOX); // (TMA,REST)
  static_assert(size<1>(tOsO) == 1);

  int lane_predicate = cute::elect_one_sync();
  // Issue the TMA store.
  if (leaderWarp and lane_predicate) {
    cute::copy(tmaStoreO, tOsO, tOgO);
  }
  
  // Wait for TMA store to complete.
  tma_store_wait<0>();

// Write out rowMax and rowSum to GMEM.
// Required for verification ONLY.
#ifdef COPYOUTMI
  Tensor miGlobal = make_tensor(make_gmem_ptr(mi_ptr), gmemLayoutMi);
  Tensor miGlobalOut =
      local_tile(miGlobal, make_shape(get<0>(tileShapeO), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  Tensor sPrimeGlobal = make_tensor(make_gmem_ptr(sPrimePtr), gmemLayoutMi);
  Tensor sPrimeGlobalOut =
      local_tile(sPrimeGlobal, make_shape(get<0>(tileShapeO), 1, 1),
                 make_coord(blockIdxX, blockIdxH, blockIdxB));
  if (threadIdx.x % 4 == 0) {
    auto mmaThreadLayoutC = TiledMma0{}.get_layoutC_TV();
    auto mmaShapeMNK = cute::tile_shape(TiledMma0{});
    auto mmaShapeMN = make_shape(shape<0>(mmaShapeMNK), shape<1>(mmaShapeMNK));
    auto flatCoord =
        cute::idx2crd(mmaThreadLayoutC(threadIdx.x, 0), mmaShapeMN);
    auto rowIdGlobal = get<0>(flatCoord); // starting rowId
    auto rowId = 0;
    for (int i = rowIdGlobal; i < kQueriesPerBlock;
         i += NumMmaWarpGroups * 64) {
      miGlobalOut(i) = rowMax(rowId);
      sPrimeGlobalOut(i) = rowSum(rowId);
      miGlobalOut(i + 8) = rowMax(rowId + 1);
      sPrimeGlobalOut(i + 8) = rowSum(rowId + 1);
      rowId += 2;
    }
  }
#endif
}
