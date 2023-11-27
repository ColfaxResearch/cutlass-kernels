namespace cfk
{
namespace utils
{
void set_smem_size(int smem_size, void const* kernel)
{
        // account for dynamic smem capacity if needed
    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        std::cout << "  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result);
      }
    }
}
}
}
