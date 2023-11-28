#!/usr/bin/bash

QBLKSIZES=( 64 128 )
KBLKSIZES=( 64 128 )
HEADSIZES=( 64 128 )
VERIFY="verify"
VERIFYFLAGS=
REFCHECK=false
INPUT1=$1
echo $1
if [[ $INPUT1 == $VERIFY ]];
then
VERIFYFLAGS="-DCOPYOUTMM0 -DCOPYOUTMI"
REFCHECK=true
fi

for QBLK in "${QBLKSIZES[@]}"
do
for KBLK in "${KBLKSIZES[@]}"
do
for HEAD in "${HEADSIZES[@]}"
do

echo "/usr/local/cuda-12.2/bin/nvcc -ccbin=/usr/bin/clang++ $VERIFYFLAGS -use_fast_math -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_CUBLAS=1 -DFMHA -DQBLKSIZE=$QBLK -DKBLKSIZE=$KBLK -DHEAD=$HEAD   -I../../lib/ -I ../../include  -I/home/bikshang/repos/cutlass-3.3/cutlass/include -I/home/bikshang/repos/cutlass-3.3/cutlass/examples/common -I"/usr/local/cuda-12.2/include" -I/include -I/examples -I/home/bikshang/repos/cutlass-3.3/cutlass/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a]  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xnvlink=--verbose -Xptxas=--verbose -std=c++17 -MD -MT -MF -x cu  fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward"

/usr/local/cuda-12.2/bin/nvcc -ccbin=/usr/bin/clang++ $VERIFYFLAGS -use_fast_math -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_CUBLAS=1 -DFMHA -DQBLKSIZE=$QBLK -DKBLKSIZE=$KBLK -DHEAD=$HEAD   -I../../lib/ -I ../../include  -I/home/bikshang/repos/cutlass-3.3/cutlass/include -I/home/bikshang/repos/cutlass-3.3/cutlass/examples/common -I"/usr/local/cuda-12.2/include" -I/include -I/examples -I/home/bikshang/repos/cutlass-3.3/cutlass/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a]  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xnvlink=--verbose -Xptxas=--verbose -std=c++17 -MD -MT -MF -x cu  fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward

./fmha_forward --batch-size=4 --seq-length=4096 --dim-size=2048 --iterations=1000  --reference-check=$REFCHECK

done
done
done

