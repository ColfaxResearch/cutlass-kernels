# FMHA Pipeline

Implementation of FMHA with software pipelining and optional warp specialization. Uses the CUTLASS Pipeline API.

## Compiler flags

You can tune a number of parameters through compiler flags. For integer flags:

1. STAGECOUNT=#stages used during pipelining. Default is 2. This must be set to a number greater than 1. Since SMEM usage is determined dynamically, too large a value may lead to "Shared Memory Allocation Failed" error at runtime.
2. CLUSTERN=#threadblocks put in a cluster. Default is 1.
3. QBLKSIZE and KBLKSIZE remain as before. QBLKSIZE is the tile length bM along the first dimension for Q, and KBLKSIZE is the tile length bN along the first dimension for K and V.

For #ifdef flags:

1. WSPL enables warp specialization.
2. CTA256 enables 256 threads (=2 warpgroups) per WGMMA.
3. GEMM1FP16ACC enables FP16 accumulator for GEMM-I. Not recommended due to severely degraded accuracy of computation (we want float accumulator for softmax).
4. GEMM2FP16ACC enables FP16 accumulator for GEMM-II. Enabling matches standard implementations of FlashAttention-2 (e.g., Tri Dao's version).
5. QINRMEM enables Q operand in RMEM for GEMM-I.
6. SINSMEM enables softmax(S) operand in SMEM for GEMM-II. Enabling this will always lead to decreased performance.

The debugging flags COPYOUTMM0 and COPYOUTMI remain as before. 

## Changes in the scripts

The compile.sh script now accepts 7 arguments as "-D$1 -D$2 -D$3 -D$4 -D$5 -D$6 -D$7". GEMM2FP16ACC is enabled by default. If no flag is desired, simply input "NONE" as the argument.

The compile_run_all_config.sh script accepts 7 arguments as follows:
1. $1 is meant to enable validation via "VERIFY" or "VERIFYALL".
2. $2 is used for "prec-type". This is 1 for FP16 and 2 for FP8.
3. $3 should be CTA256 or NONE.
4. $4 through $7 are additional optional flags: we use them for STAGECOUNT, CLUSTERN, WSPL, and QINRMEM.

The script then runs over predetermined possibilities for QBLKSIZE, KBLKSIZE, and head dimension 64, 128, and 256.

## NOTES:

1. Tested with CUDA 12.2, CUTLASS 3.3 and SM90A.