# CUTLASS Kernels

Library of CUTLASS kernels targeting Large Language Models (LLM).

# Building and Running.

1. Download CUTLASS following instructions from: https://github.com/NVIDIA/cutlass.
2. Modify the (hardcoded) path in the sample compile_run_all_config.sh to your CUTLASS directory.
3. Run the modified compile.sh as ./compile_run_all_config.sh [verify]. 
4. The script also runs for pre-defined arguments.

# Building and Running for custom tile shapes and head dim.

1. To run for different Tiling configurations, the script shall be modified to use different shapes for QBLK and KBLK, where
QBLK is the QUERY_BLOCK_SIZE and KBLK is the KEY_BLOCK_SIZE. HEADDIM can be set to different values too.

2. To run "fmha_forward", run "fmha_forward --help" first to get list of command line arguments to be supplied for the executable.
 
