# FMHA Pipeline

Implementation of FMHA with software pipelining and optional warp specialization. Uses the CUTLASS Pipeline API. Both FP16 and FP8 precision formats for input tensors are supported, with the FP8 version implemented as a 'hybrid' kernel where the Q and K tensors are FP8 but the V tensor remains FP16. We note that this choice agrees with the FP8 implementation from the Triton FMHA tutorial.

## Compiler flags

You can tune a number of parameters through compiler flags. For integer flags:

1. STAGECOUNT=#stages used during pipelining. Default is 2. This must be set to a number greater than 1. Since SMEM usage is determined dynamically, too large a value may lead to "Shared Memory Allocation Failed" error at runtime.
2. CLUSTERN=#threadblocks put in a cluster. Default is 1.
3. QBLKSIZE is the tile length bM along the first dimension for Q, and KBLKSIZE is the tile length bN along the first dimension for K and V. Default is 64 for both. We have tested for {64,128}x{64,128}. When CTA256 is enabled, QBLKSIZE should be 128.

For #ifdef flags:

1. WSPL enables warp specialization.
2. CTA256 enables 256 threads (=2 warpgroups) per WGMMA.
3. QINRMEM enables Q operand in RMEM for GEMM-I.
4. GEMM1FP16ACC enables FP16 accumulator for GEMM-I. Not recommended due to severely degraded accuracy of computation (we want float accumulator for softmax).
5. GEMM2FP16ACC enables FP16 accumulator for GEMM-II.

The debugging flags COPYOUTMM0 and COPYOUTMI remain as before. 

## Building and Running with the scripts

1. Download CUTLASS 3.4 following instructions from: https://github.com/NVIDIA/cutlass.
2. Change the hardcoded paths in the 'compile.sh' and 'compile_run_all_config.sh' scripts to your CUTLASS directory.

The 'compile.sh' script accepts 7 arguments (as "-D$1 -D$2 -D$3 -D$4 -D$5 -D$6 -D$7"). If no flag is desired, simply input "NONE" as the argument.

The 'compile_run_all_config.sh' script accepts 7 arguments as follows:

1. $1 is meant to enable validation via "VERIFY" or "VERIFYALL".
2. $2 is used for "prec-type". This is 1 for FP16 and 2 for FP8.
3. $3 should be CTA256 or NONE.
4. $4 through $7 are additional optional flags: we use them for STAGECOUNT, CLUSTERN, WSPL, and QINRMEM.

The script then runs over predetermined possibilities for QBLKSIZE, KBLKSIZE, and head dimensions 64, 128, and 256.

## NOTES:

1. Tested with CUDA 12.2/12.3, CUTLASS 3.3/3.4 and SM90A.
2. A small change in the CUTLASS Pipeline API from 3.3 to 3.4 necessitates changes in the code for successful compilation. This version compiles with 3.4 and we have indicated the small changes required for use with 3.3 in the comments of 'fmha_driver.h' and 'fmha_driver_ws.h'.
3. A choice of large tile size may lead to nvcc warnings related to register spilling. However, since we support head dimensions {64,128,256} and {FP16,FP8} within the same program, one should be observant as to which choice(s) of head dimension and precision type is triggering the warning. This is easily seen from the verbose ptxas output about register usage.
