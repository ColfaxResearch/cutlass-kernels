# CUTLASS Kernels

Library of CUTLASS kernels targetting Large Language Models (LLM).

# Building 

1. Downlaod CUTLASS following instructions from: https://github.com/NVIDIA/cutlass.
2. Modify the (hardcoded) path in the sample compile.sh to your CUTLASS directory.
3. Run the modified compile.sh as ./compile.sh.

# Running

1. While running the executable make sure to set NVIDIA_TF32_OVERRIDE=1 to enable TF32 mode for cuBLAS for SGEMM. Otherwise, cuBLAS uses float32.
