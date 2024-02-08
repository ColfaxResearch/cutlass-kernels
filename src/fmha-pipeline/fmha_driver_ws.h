#pragma once
/*! \file
    \brief Unit test for the PipelineTmaAsync class as it would be used in a
   Warp specialized loop
*/

#define KERNEL_DBG_TRACE false

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>

#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/reference/host/gemm.h>

#include "cutlass/core_io.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "fmha_consumer.h"
#include "fmha_producer.h"
#include "shared_storage.h"

using namespace cute;
using namespace cutlass;

// Main FMHA Device Kernel.
// Gemm1Type = Precision of Computation used by GEMM (half_t by default).
// SoftType = Type of Accumulator used by GEMM (float by default).
// Other types are self-explanatory.
template <class Gemm1Type, class AccumType, class SoftType, class Gemm2Type,
          class OutputType, class TiledMma0, class TiledMma1, class TiledCopyQ,
          class TileShapeQ, class GmemLayoutQ, class SmemLayoutQ,
          class TiledCopyK, class TileShapeK, class GmemLayoutK,
          class SmemLayoutK, class TileShapeS, class GmemLayoutS,
          class SmemLayoutS, class TiledCopyV, class TileShapeV,
          class GmemLayoutV, class SmemLayoutV, class SmemLayoutVt,
          class SrcSmemLayoutV, class SrcSmemLayoutAtomV, class TiledCopyO,
          class TileShapeO, class GmemLayoutO, class SmemLayoutO,
          class GmemLayoutMI, class ClusterShape>
__global__ static void __launch_bounds__(384, 1) fmhaForwardWS(
    Gemm1Type const *Q, CUTE_GRID_CONSTANT TiledCopyQ const tmaLoadQ,
    TileShapeQ tileShapeQ, GmemLayoutQ gmemLayoutQ, SmemLayoutQ smemLayoutQ,
    Gemm1Type const *K, CUTE_GRID_CONSTANT TiledCopyK const tmaLoadK,
    TileShapeK tileShapeK, GmemLayoutK gmemLayoutK, SmemLayoutK smemLayoutK,
    Gemm1Type *S, TileShapeS tileShapeS, GmemLayoutS gmemLayoutS,
    SmemLayoutS smemLayoutS, int nTilesOfK, Gemm2Type *V,
    CUTE_GRID_CONSTANT TiledCopyV const tmaLoadV, TileShapeV tileShapeV,
    GmemLayoutV gmemLayoutV, SmemLayoutV smemLayoutV, SmemLayoutVt smemLayoutVt,
    SrcSmemLayoutV srcSmemLayoutV, SrcSmemLayoutAtomV srcSmemLayoutAtomV,
    OutputType *O, CUTE_GRID_CONSTANT TiledCopyO const tmaStoreO,
    TileShapeO tileShapeO, GmemLayoutO gmemLayoutO, SmemLayoutO smemLayoutO,
    SoftType *mi_ptr, SoftType *sPrimePtr, GmemLayoutMI gmemLayoutMi,
    float scale)

{
  extern __shared__ char shared_memory[];
  using MainloopPipeline =
      typename cutlass::PipelineTmaAsync<stageCount, ClusterShape>;
  using PipelineState = typename cutlass::PipelineState<stageCount>;

  using SharedStorage =
      SharedStorage<Gemm1Type, Gemm2Type, OutputType, SmemLayoutQ, SmemLayoutK,
                    SmemLayoutS, SmemLayoutV, SmemLayoutO, ClusterShape>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  int warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  int warp_idx_in_warpgroup =
      __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
  int warp_group_thread_idx = threadIdx.x % 128;
  dim3 block_id_in_cluster = cute::block_id_in_cluster();

  auto cluster_shape = ClusterShape{};

  // TODO: #Producers = #RowsInCluster + #ColsInCluster - 1
  uint32_t const NumProducers = 1;

  // Get the full un-partitioned tensors.
  // TMA tensors are special tensors.
  Tensor mQ = tmaLoadQ.get_tma_tensor(shape(gmemLayoutQ));
  Tensor mK = tmaLoadK.get_tma_tensor(shape(gmemLayoutK));
  Tensor mV = tmaLoadV.get_tma_tensor(shape(gmemLayoutV));
  Tensor gK = local_tile(mK, tileShapeK, make_coord(0, 0, 0, 0));
  Tensor gV = local_tile(mV, tileShapeV, make_coord(0, 0, 0, 0));
  constexpr int per_cta_bytes = size(gK) * sizeof_bits_v<Gemm1Type> / 8 +
                                size(gV) * sizeof_bits_v<Gemm2Type> / 8;
  uint32_t const TmaTransactionBytes = per_cta_bytes * NumProducers;

  // Construct SMEM tensors.
  Tensor sQ =
      make_tensor(make_smem_ptr(shared_storage.qo.smem_q.data()), smemLayoutQ);
  // Reuse sQ for sO (with new smem layout for FP8).
  Tensor sO =
      make_tensor(make_smem_ptr(shared_storage.qo.smem_o.data()), smemLayoutO);
  Tensor sK =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_k.data()), smemLayoutK);
#ifdef SINSMEM
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_s.data()), smemLayoutS);
#else
  // Just a dummy sS (with smem_v). It's required only for shape later.
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_k.data()), smemLayoutS);
#endif
  Tensor sV =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_v.data()), smemLayoutV);

  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_v.data()), smemLayoutVt);

  Tensor mO = make_tensor(make_gmem_ptr(O), gmemLayoutO);

  TiledMma0 tiledMma0;
  TiledMma1 tiledMma1;
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);

  //
  // Partition the copying of dest tiles for Q, K and V among threads.
  //

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
  Tensor tOrS = threadMma1.partition_fragment_A(sS(_, _, 0));
  auto tOrPLayout = ReshapeTStoTP()(tSrS, tOrS);

  // No pipeling for copying the block of Q.

  // Get the block of Q for this CTA using the block coordinates
  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);
  auto blkCoordQ = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gQ = local_tile(mQ, tileShapeQ, blkCoordQ);

  // Partition the copying of source tiles for Q among threads.
  auto cta_tmaQ = tmaLoadQ.get_slice(0);
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
  Tensor tQsQ = group_modes<1, rank(tQsQX)>(tQsQX);
  // Copy Q tile from GMEM to SMEM.
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;
  cfk::barrierInit(tma_load_mbar[0], 1); // This is REQUIRED.
  cfk::copy(tQgQ(_, 0), tQsQ(_, 0), tmaLoadQ, tma_load_mbar[0]);
  cute::wait_barrier(tma_load_mbar[0], 0); // This is REQUIRED.

#ifdef QINRMEM
  Tensor tSsQ = threadMma0.partition_A(sQ);
  cfk::copy(tSsQ, tSrQ);
#endif

  // FMHA OUTPUT (GEMM-II) accumulator.
  Tensor tOrO = partition_fragment_C(tiledMma1, tileShapeO);
  clear(tOrO);
  // Allocate space for per-thread rowMax and rowSum in rmem.
  Tensor rowMax = make_tensor<SoftType>(Shape<Int<2 * size<1>(tSrS)>>{});
  Tensor rowSum = make_fragment_like(rowMax);
  cute::fill(rowMax, -cutlass::platform::numeric_limits<SoftType>::infinity());
  cute::fill(rowSum, SoftType(0.0));
  // ------------ Pipelining begins -------------------------------

  // mbarrier.init
  typename MainloopPipeline::Params params;
  params.transaction_bytes = TmaTransactionBytes;
  if (warp_group_idx == 0) {
    params.role = MainloopPipeline::ThreadCategory::Producer;
  } else {
    params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  params.is_leader = warp_group_thread_idx == 0;

  params.num_consumers = NumMmaThreads;

  MainloopPipeline pipeline(shared_storage.storage, params);

  int blockIdxY = 0;

  __syncthreads();

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  // Producer WarpGroup
  if (warp_group_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<40>();

    int lane_predicate = cute::elect_one_sync();
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {

      int tma_k_prologue = min(stageCount, nTilesOfK);

      // Simulating Prologue TMA Loads
      // For the DMA (prologue) - we start with an opposite phase - since we
      // skip all waits i.e., we know that the buffer is indeed empty
      PipelineState smem_pipe_write =
          make_producer_start_state<MainloopPipeline>();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tma_k_prologue; ++i) {
        pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
        fmhaForwardProducer(sK(_, _, i), tmaLoadK, tileShapeK, gmemLayoutK,
                            sV(_, _, i), tmaLoadV, tileShapeV, gmemLayoutV,
                            blockIdxY++, tmaBar, ClusterShape());

        // Simulating cp.async.bulk.tensor behavior
        // pipeline.producer_commit(smem_pipe_write, per_cta_bytes);
        ++smem_pipe_write;
      }
      int tma_k_iter = nTilesOfK - tma_k_prologue;

      // Simulating Mainloop TMA Loads
      CUTE_NO_UNROLL
      for (; tma_k_iter > 0; --tma_k_iter) {

        pipeline.producer_acquire(smem_pipe_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
        auto stage = smem_pipe_write.index();
        fmhaForwardProducer(sK(_, _, stage), tmaLoadK, tileShapeK, gmemLayoutK,
                            sV(_, _, stage), tmaLoadV, tileShapeV, gmemLayoutV,
                            blockIdxY++, tmaBar, ClusterShape());

        // Simulating cp.async.bulk.tensor behavior
        // pipeline.producer_commit(smem_pipe_write, per_cta_bytes);

        // Advance write stage
        ++smem_pipe_write;
      }

      // Tail Loop
      // Handles the case where we never enter the mainloop
      PipelineState tail =
          tma_k_prologue == stageCount ? smem_pipe_write : PipelineState{};
      for (int i = 0; i < tma_k_prologue; ++i) {
        pipeline.producer_acquire(tail);
        ++tail;
      }
    }
    // Consumer WarpGroup
  } else if (warp_group_idx == 1 || warp_group_idx == 2) {
    cutlass::arch::warpgroup_reg_alloc<232>();

    // print("consumer yes\n");
    PipelineState smem_pipe_read;
    PipelineState smem_pipe_release;

    // Init Shared Memory read stages & PhaseBit
    static constexpr uint32_t K_PIPE_MMAS = 1;
    static_assert(K_PIPE_MMAS < stageCount, "ERROR : Too many MMAs in flight");

    // Total number of gemm iterations
    auto gemm_k_iterations = nTilesOfK;

    // Simulating Prologue MMAs
    int mma_k_prologue = min(K_PIPE_MMAS, gemm_k_iterations);
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < mma_k_prologue; ++iter) {
      pipeline.consumer_wait(smem_pipe_read);

      warpgroup_arrive();

#if 1
      int stage = smem_pipe_read.index();

      fmhaForwardConsumer(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                          tOrV(_, _, _, stage), tOrO, tOrPLayout, rowMax,
                          rowSum, tileShapeS, gmemLayoutS, scale, blockIdxY++,
                          tiledMma0, tiledMma1, AccumType(0), SoftType(0));
#endif

      ++smem_pipe_read;
    }
    gemm_k_iterations -= mma_k_prologue;

    // Simulating Mainloop MMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {

      /// Wait on the smem_pipe_read stage / phase
      pipeline.consumer_wait(smem_pipe_read);

      warpgroup_arrive();

#if 1
      int stage = smem_pipe_read.index();

      fmhaForwardConsumer(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                          tOrV(_, _, _, stage), tOrO, tOrPLayout, rowMax,
                          rowSum, tileShapeS, gmemLayoutS, scale, blockIdxY++,
                          tiledMma0, tiledMma1, AccumType(0), SoftType(0));

      warpgroup_wait<2 * K_PIPE_MMAS>();
//    warpgroup_fence_operand(tSrS);
//    warpgroup_fence_operand(tOrO);
#endif

      pipeline.consumer_release(smem_pipe_release);

      // Advance stages
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_wait<0>();
    // Tail Loop
    for (int i = 0; i < K_PIPE_MMAS; ++i) {
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;
    }

    bool leaderWarp = warp_group_idx == 1 && warp_idx_in_warpgroup == 0;
    fmhaForwardWriteOutTMA(tOrO, rowMax, rowSum, O, tileShapeO, gmemLayoutO,
                           mi_ptr, sPrimePtr, gmemLayoutMi, tiledMma0,
                           tiledMma1, sO, tmaStoreO, leaderWarp);
  }
}
