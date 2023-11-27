#!/usr/bin/bash

QBLKSIZES=( 64 128 )
KBLKSIZES=( 64 128 )
HEADSIZES=( 64 128 )


for QBLK in "${QBLKSIZES[@]}"
do
for KBLK in "${KBLKSIZES[@]}"
do
for HEAD in "${HEADSIZES[@]}"
do

/usr/local/cuda-12.2/bin/nvcc -DNOVERIFY  -ccbin=/usr/bin/clang++ -use_fast_math -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_CUBLAS=1 -DFMHA -DQBLKSIZE=$QBLK -DKBLKSIZE=$KBLK -DHEAD=$HEAD   -I../../lib/ -I ../../include  -I/home/bikshang/repos/cutlass-3.3/cutlass/include -I/home/bikshang/repos/cutlass-3.3/cutlass/examples/common -I"/usr/local/cuda-12.2/include" -I/include -I/examples -I/home/bikshang/repos/cutlass-3.3/cutlass/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a]  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xnvlink=--verbose -Xptxas=--verbose -std=c++17 -MD -MT -MF -x cu  fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward_opt_fmha

NVIDIA_TF32_OVERRIDE=1  ./fmha_forward_opt_fmha 2 4096 4096 2048 4 1000 10

done
done
done


#TF32 use SINSMEM.
for QBLK in "${QBLKSIZES[@]}"
do
for KBLK in "${KBLKSIZES[@]}"
do
for HEAD in "${HEADSIZES[@]}"
do
/usr/local/cuda-12.2/bin/nvcc -DNOVERIFY -DCTA256 -ccbin=/usr/bin/clang++ -use_fast_math -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_CUBLAS=1 -DSINSMEM -DFMHA -DQBLKSIZE=$QBLK -DKBLKSIZE=$KBLK -DHEAD=$HEAD   -I../../lib/ -I ../../include  -I/home/bikshang/repos/cutlass-3.3/cutlass/include -I/home/bikshang/repos/cutlass-3.3/cutlass/examples/common -I"/usr/local/cuda-12.2/include" -I/include -I/examples -I/home/bikshang/repos/cutlass-3.3/cutlass/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a] -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xnvlink=--verbose -Xptxas=--verbose -std=c++17 -MD -MT -MF -x cu  fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward_opt_fmha

NVIDIA_TF32_OVERRIDE=1  ./fmha_forward_opt_fmha 2 4096 4096 2048 4 1000 10
done
done
done
