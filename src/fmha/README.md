# FMHA

An implementation of Flash Multi Head Attention (FMHA) algorithm using CUTLASS/CuTe for SM90.

# Building and Running.

1. Download CUTLASS following instructions from: https://github.com/NVIDIA/cutlass.
2. Modify the (hardcoded) path in the sample compile.sh to your CUTLASS directory.
3. To run "fmha_forward", run "fmha_forward --help" first to get list of command line arguments to be supplied for the executable.

# Building and Running for custom tile shapes.

1. Change the hardcoded path in  ./compile_run_all_config.sh to your CUTLASS directory.

2. Run as  ./compile_run_all_config.sh [verify].  The script runs for pre-defined tile shapes for QBLK and BLK, where
QBLK is the QUERY_BLOCK_SIZE and KBLK is the KEY_BLOCK_SIZE.

3. To run for different Tiling configurations, the script shall be modified to use different shapes for QBLK and KBLK.
