#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>

#include "cute/arch/cluster_sm90.hpp"

using namespace cute;

namespace cfk {
template <typename TA, typename LayoutA, typename TB, typename LayoutB,
          typename TC, typename LayoutC, typename TiledMma>
__device__ void gemm(TiledMma &tiled_mma, const Tensor<TA, LayoutA> &tCrA,
                     const Tensor<TB, LayoutB> &tCrB,
                     Tensor<TC, LayoutC> &tCrC) {
  auto num_k = size<2>(tCrA);
  warpgroup_fence_operand(tCrC);

  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

  warpgroup_commit_batch();
  warpgroup_wait<1>();
  __syncthreads();
}
} // namespace cfk
