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
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();

  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);
  __syncthreads();
}

template <typename TA, typename LayoutA, typename TB, typename LayoutB,
          typename TC, typename LayoutC, typename TiledMma>
__device__ void gemm_bar_nowait(TiledMma &tiled_mma, const Tensor<TA, LayoutA> &tCrA,
                     const Tensor<TB, LayoutB> &tCrB,
                     Tensor<TC, LayoutC> &tCrC,
                     cute::uint64_t & tma_load_mbar){
  /// Wait on the shared memory barrier until the phase bit flips from
  /// kPhaseBit value
  constexpr int kPhaseBit = 0;
  cute::wait_barrier(tma_load_mbar, kPhaseBit);
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();

  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

}


template <typename TA, typename LayoutA, typename TB, typename LayoutB,
          typename TC, typename LayoutC, typename TiledMma>
__device__ void gemm_bar_wait(TiledMma &tiled_mma, const Tensor<TA, LayoutA> &tCrA,
                     const Tensor<TB, LayoutB> &tCrB,
                     Tensor<TC, LayoutC> &tCrC,
                     cute::uint64_t & tma_load_mbar){
  /// Wait on the shared memory barrier until the phase bit flips from
  /// kPhaseBit value
  constexpr int kPhaseBit = 0;
  cute::wait_barrier(tma_load_mbar, kPhaseBit);
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();

  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);
  __syncthreads();
}

template <typename TA, typename LayoutA, typename TB, typename LayoutB,
          typename TC, typename LayoutC, typename TiledMma, typename TransposeFn,
          typename TensorB, typename TensorBt>
__device__ void gemm_bar_wait_transB(TiledMma &tiled_mma, const Tensor<TA, LayoutA> &tCrA,
                     const Tensor<TB, LayoutB> &tCrB,
                     Tensor<TC, LayoutC> &tCrC,
                     cute::uint64_t & tma_load_mbar,
                     TransposeFn & transposeFn,
                     const TensorB & sB,
                     const TensorBt & sBt){
  /// Wait on the shared memory barrier until the phase bit flips from
  /// kPhaseBit value
  constexpr int kPhaseBit = 0;
  cute::wait_barrier(tma_load_mbar, kPhaseBit);
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();

#if 0
  if (thread0()) 
  {
       print ("B\n");
       print_tensor(sBt);
  }
#endif

  transposeFn.transpose(sB, sBt, 0);
  transposeFn.synchronize();

#if 0
  if (thread0()) 
  {
       print ("Bt\n");
       print_tensor(sBt);
  }
#endif

  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);
  __syncthreads();
}

} // namespace cfk
