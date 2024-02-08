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
__global__ static void //__launch_bounds__(256, 1)
fmhaForward(Gemm1Type const *Q, CUTE_GRID_CONSTANT TiledCopyQ const tmaLoadQ,
            TileShapeQ tileShapeQ, GmemLayoutQ gmemLayoutQ,
            SmemLayoutQ smemLayoutQ, Gemm1Type const *K,
            CUTE_GRID_CONSTANT TiledCopyK const tmaLoadK, TileShapeK tileShapeK,
            GmemLayoutK gmemLayoutK, SmemLayoutK smemLayoutK, Gemm1Type *S,
            TileShapeS tileShapeS, GmemLayoutS gmemLayoutS,
            SmemLayoutS smemLayoutS, int nTilesOfK, Gemm2Type *V,
            CUTE_GRID_CONSTANT TiledCopyV const tmaLoadV, TileShapeV tileShapeV,
            GmemLayoutV gmemLayoutV, SmemLayoutV smemLayoutV,
            SmemLayoutVt smemLayoutVt, SrcSmemLayoutV srcSmemLayoutV,
            SrcSmemLayoutAtomV srcSmemLayoutAtomV, OutputType *O,
            CUTE_GRID_CONSTANT TiledCopyO const tmaStoreO,
            TileShapeO tileShapeO, GmemLayoutO gmemLayoutO,
            SmemLayoutO smemLayoutO, SoftType *mi_ptr, SoftType *sPrimePtr,
            GmemLayoutMI gmemLayoutMi, float scale)

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

  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
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
  // Re-use sQ for sO too but with different layout (for fp8).
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
  cfk::barrierInit(tma_load_mbar[0], 1);
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

  // if (cute::thread0())
  //  print("yes\n");

  // mbarrier.init
  typename MainloopPipeline::Params params;
  params.transaction_bytes = TmaTransactionBytes;
  params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
  params.is_leader = warp_group_thread_idx == 0;

  params.num_consumers = NumMmaThreads;

  MainloopPipeline pipeline(shared_storage.storage, params);

  int blockIdxYProd = 0;
  int blockIdxYCons = 0;

  __syncthreads();

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  // DMA Prologue (Loads)
  // Total number of gemm_k_iterations
  auto mma_k_iterations = nTilesOfK;
  auto tma_k_iterations = nTilesOfK;

  PipelineState smem_pipe_read;
  // For the DMA (prologue) - we start with an opposite phase - since we skip
  // all waits i.e., we know that the buffer is indeed empty
  PipelineState smem_pipe_write =
      cutlass::make_producer_start_state<MainloopPipeline>();
  PipelineState smem_pipe_release;
  int K_TILE_MMAS = 1;

  int k_pipe_tma_prologue = min(stageCount, tma_k_iterations);

  int lane_predicate = cute::elect_one_sync();
  if (warp_idx == 0 && lane_predicate) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < k_pipe_tma_prologue; ++i) {
      pipeline.producer_acquire(smem_pipe_write);
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
      fmhaForwardProducer(sK(_, _, i), tmaLoadK, tileShapeK, gmemLayoutK,
                          sV(_, _, i), tmaLoadV, tileShapeV, gmemLayoutV,
                          blockIdxYProd++, tmaBar, ClusterShape());
      ++smem_pipe_write;
    }
  }
  tma_k_iterations -= k_pipe_tma_prologue;

  // MMA Prologue (Compute) - modeling inflight MMAs
  for (int iter = 0; iter < K_TILE_MMAS; ++iter) {
    pipeline.consumer_wait(smem_pipe_read);
    warpgroup_arrive();
    int stage = smem_pipe_read.index();

    fmhaForwardConsumer(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                        tOrV(_, _, _, stage), tOrO, tOrPLayout, rowMax, rowSum,
                        tileShapeS, gmemLayoutS, scale, blockIdxYCons++,
                        tiledMma0, tiledMma1, AccumType(0), SoftType(0));

    ++smem_pipe_read;
  }

  mma_k_iterations -= K_TILE_MMAS;

  CUTLASS_PRAGMA_NO_UNROLL
  for (int iter = 0; iter < mma_k_iterations; ++iter) {
    pipeline.consumer_wait(smem_pipe_read);

    warpgroup_arrive();
    // GMMA would typically happen here
    int stage = smem_pipe_read.index();

    fmhaForwardConsumer(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                        tOrV(_, _, _, stage), tOrO, tOrPLayout, rowMax, rowSum,
                        tileShapeS, gmemLayoutS, scale, blockIdxYCons++,
                        tiledMma0, tiledMma1, AccumType(0), SoftType(0));

    pipeline.consumer_release(smem_pipe_release);

    if (lane_predicate && (warp_idx == 0) && (tma_k_iterations > 0)) {
      pipeline.producer_acquire(smem_pipe_write);
      // cp.async.bulk.tensor would typically happen here
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType *tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
      auto stage = smem_pipe_write.index();
      fmhaForwardProducer(sK(_, _, stage), tmaLoadK, tileShapeK, gmemLayoutK,
                          sV(_, _, stage), tmaLoadV, tileShapeV, gmemLayoutV,
                          blockIdxYProd++, tmaBar, ClusterShape());

      ++smem_pipe_write;
      --tma_k_iterations;
    }

    // next read stage
    ++smem_pipe_read;
    ++smem_pipe_release;
  }

  warpgroup_wait<0>();

  bool leaderWarp = warp_idx == 0;
  fmhaForwardWriteOutTMA(tOrO, rowMax, rowSum, O, tileShapeO, gmemLayoutO,
                         mi_ptr, sPrimePtr, gmemLayoutMi, tiledMma0, tiledMma1,
                         sO, tmaStoreO, leaderWarp);
  // To make sure remote SMEM doesn't get destoryed
  cute::cluster_arrive();
  cute::cluster_wait();
}
