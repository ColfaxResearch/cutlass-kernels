#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include <cute/tensor.hpp>

template <typename Fragment>
CUTLASS_DEVICE static void reorgCFp8toAFp8(Fragment &accum) {

  using namespace cute;
  auto laneId = cutlass::canonical_lane_idx();

  // First update `mi` to the max per-row
  //
  auto VT = shape<0>(accum); // number of vector elements per tile.
  auto MT = shape<1>(accum); // number of tiles along M.
  auto NT = shape<2>(accum); // number of tiles along N.

  auto data = accum.data();
  int n = 0;

  auto reduceMask = 0x66666666;
#pragma unroll
  for (int i = 0; i < MT; ++i) {

    // Traverse 2-rows + 2-cols (2x2) simultaneously.

#pragma unroll
    for (int k = 0; k < NT * size<2>(VT) / 2; ++k) {

      // d0, d1, d2, d3.
      int32_t upper;
      cutlass::float_e4m3_t* byte0 = (cutlass::float_e4m3_t *)&upper + 0;
      cutlass::float_e4m3_t* byte1 = (cutlass::float_e4m3_t *)&upper + 1;
      cutlass::float_e4m3_t* byte2 = (cutlass::float_e4m3_t *)&upper + 2;
      cutlass::float_e4m3_t* byte3 = (cutlass::float_e4m3_t *)&upper + 3;

      // d4, d5, d6, d7.
      int32_t lower;
      cutlass::float_e4m3_t* byte4 = (cutlass::float_e4m3_t *)&lower + 0;
      cutlass::float_e4m3_t* byte5 = (cutlass::float_e4m3_t *)&lower + 1;
      cutlass::float_e4m3_t* byte6 = (cutlass::float_e4m3_t *)&lower + 2;
      cutlass::float_e4m3_t* byte7 = (cutlass::float_e4m3_t *)&lower + 3;

	  *byte0 = data[n];
	  *byte1 = data[n+1];
	  *byte2 = data[n+2];
	  *byte3 = data[n+3];
	  *byte4 = data[n+4];
	  *byte5 = data[n+5];
	  *byte6 = data[n+6];
	  *byte7 = data[n+7];

	  int32_t* exVal0;
      if (laneId %  4 == 0 || laneId %  4 == 1 ) {
        exVal0 = &lower;
      } else {
        exVal0 = &upper;
      }

      *exVal0 = __shfl_xor_sync(uint32_t(-1), *exVal0, 2);

      int32_t* exVal1;
      if (laneId % 2 == 0) {
        exVal1 = &lower;
      } else {
        exVal1 = &upper;
      }

      *exVal1 = __shfl_xor_sync(uint32_t(-1), *exVal1, 1);

	  data[n++] = *byte0;
	  data[n++] = *byte1;
	  data[n++] = *byte4;
      data[n++] = *byte5;

      data[n++] = *byte2;
      data[n++] = *byte3;
      data[n++] = *byte6;
      data[n++] = *byte7;
    }
  }
}

template <typename Fragment>
CUTLASS_DEVICE static void reorgCFp32toAFp8(Fragment &accum) {

  using namespace cute;
  auto laneId = cutlass::canonical_lane_idx();

  // First update `mi` to the max per-row
  //
  auto VT = shape<0>(accum); // number of vector elements per tile.
  auto MT = shape<1>(accum); // number of tiles along M.
  auto NT = shape<2>(accum); // number of tiles along N.

  auto data = accum.data();
  int n = 0;

  auto reduceMask = 0x66666666;
#pragma unroll
  for (int i = 0; i < MT; ++i) {

    // Traverse 2-rows + 2-cols (2x2) simultaneously.

#pragma unroll
    for (int k = 0; k < NT * size<2>(VT) / 2; ++k) {
      auto d0 = data[n];
      auto d1 = data[n + 1];
      auto d2 = data[n + 2];
      auto d3 = data[n + 3];

      auto d4 = data[n + 4];
      auto d5 = data[n + 5];
      auto d6 = data[n + 6];
      auto d7 = data[n + 7];

      // exhange d4, d5 with d0, d1.
      float val00 = 0;
      float val01 = 0;
      float val10 = 0;
      float val11 = 0;
      int srcLane = 0;
      if (laneId % 2 == 0) {
        val00 = d4;
        val01 = d5;
        val10 = d6;
        val11 = d7;
        srcLane = laneId + 1;
      } else {
        val00 = d0;
        val01 = d1;
        val10 = d2;
        val11 = d3;
        srcLane = laneId - 1;
      }
      val00 = __shfl_sync(uint32_t(-1), val00, srcLane);
      val01 = __shfl_sync(uint32_t(-1), val01, srcLane);
      val10 = __shfl_sync(uint32_t(-1), val10, srcLane);
      val11 = __shfl_sync(uint32_t(-1), val11, srcLane);

      if (laneId % 2 == 0) {
        d4 = val00;
        d5 = val01;
        d6 = val10;
        d7 = val11;
      } else {
        d0 = val00;
        d1 = val01;
        d2 = val10;
        d3 = val11;
      }

      // exhange d4, d5 with d0, d1.
      srcLane = 0;
      if (laneId % 4 == 1) {
        srcLane = laneId + 1;
      } else if (laneId % 4 == 2) {
        srcLane = laneId - 1;
      }

      if (laneId % 4 == 1 || laneId % 4 == 2) {
        d0 = __shfl_sync(reduceMask, d0, srcLane);
        d1 = __shfl_sync(reduceMask, d1, srcLane);
        d2 = __shfl_sync(reduceMask, d2, srcLane);
        d3 = __shfl_sync(reduceMask, d3, srcLane);
        d4 = __shfl_sync(reduceMask, d4, srcLane);
        d5 = __shfl_sync(reduceMask, d5, srcLane);
        d6 = __shfl_sync(reduceMask, d6, srcLane);
        d7 = __shfl_sync(reduceMask, d7, srcLane);
      }

      data[n++] = d0;
      data[n++] = d1;
      data[n++] = d4;
      data[n++] = d5;

      data[n++] = d2;
      data[n++] = d3;
      data[n++] = d6;
      data[n++] = d7;
    }
  }
}

template <typename CType, typename AType, typename Fragment>
CUTLASS_DEVICE static void reorgCtoA(Fragment &accum) {

  if constexpr (is_same_v<AType, cutlass::half_t>) {
    // No reorg required; do nothing.
    return;
  } else if constexpr (is_same_v<AType, cutlass::float_e4m3_t>) {
    if constexpr (is_same_v<CType, float>) {
      reorgCFp32toAFp8(accum);
    } else if constexpr (is_same_v<CType, cutlass::float_e4m3_t>) {
      reorgCFp8toAFp8(accum);
    } else {
      static_assert(is_same_v<CType, float> ||
                    is_same_v<CType, cutlass::float_e4m3_t>);
    }
  } else {
    static_assert(is_same_v<AType, cutlass::half_t> ||
                  is_same_v<AType, cutlass::float_e4m3_t>);
  }
}
