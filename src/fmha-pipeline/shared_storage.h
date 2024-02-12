#pragma once

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

// Default stage count is 2
#ifdef STAGECOUNT
constexpr int stageCount = STAGECOUNT;
#else
constexpr int stageCount = 2;
#endif

#ifdef CTA256
constexpr int NumMmaWarpGroups = 2;
constexpr int NumMmaThreads = 256;
#else
constexpr int NumMmaWarpGroups = 1;
constexpr int NumMmaThreads = 128;
#endif

// Shared Storage with Aligned addresses.
template <class Gemm1Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutO>
union SharedStorageQO {
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
};

// Shared Storage with Aligned addresses.
template <class Gemm1Type, class Gemm2Type, class SmemLayoutK,
          class SmemLayoutS, class SmemLayoutV>
struct SharedStorageKV {
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
#ifdef SINSMEM
  cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutS>> smem_s;
#endif
};

// Shared Storage with Aligned addresses.
#ifdef QINRMEM
template <class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutS, class SmemLayoutV,
          class SmemLayoutO,
          typename ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>>
struct SharedStorage {
  union {
    SharedStorageQO<Gemm1Type, OutputType, SmemLayoutQ, SmemLayoutO> qo;
    SharedStorageKV<Gemm1Type, Gemm2Type, SmemLayoutK, SmemLayoutS, SmemLayoutV>
        kv;
  };
  struct {
    cute::uint64_t tma_load_mbar[1];
    typename cutlass::PipelineTmaAsync<stageCount>::SharedStorage
        storage;
  };
};
#else
template <class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutS, class SmemLayoutV,
          class SmemLayoutO,
          typename ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>>
struct SharedStorage {
  struct {
    SharedStorageQO<Gemm1Type, OutputType, SmemLayoutQ, SmemLayoutO> qo;
    SharedStorageKV<Gemm1Type, Gemm2Type, SmemLayoutK, SmemLayoutS, SmemLayoutV>
        kv;
  };
  struct {
    cute::uint64_t tma_load_mbar[1];
    typename cutlass::PipelineTmaAsync<stageCount>::SharedStorage
        storage;
  };
};
#endif
