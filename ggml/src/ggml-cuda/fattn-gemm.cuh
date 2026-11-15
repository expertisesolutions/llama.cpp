#pragma once

#include "common.cuh"

#if defined(GGML_USE_HIP)
// cuBLAS-GEMM flash attention is for pre-Volta NVIDIA only, not built for HIP (fattn-gemm.cu is excluded
// in ggml-hip/CMakeLists.txt). enabled() is false, so the other two are never reached.
static inline bool ggml_cuda_fa_gemm_enabled() { return false; }
static inline bool ggml_cuda_flash_attn_ext_gemm_supported(const ggml_tensor *) { return false; }
static inline void ggml_cuda_flash_attn_ext_gemm(ggml_backend_cuda_context &, ggml_tensor *) { GGML_ABORT("fattn-gemm is not built for HIP"); }
#else // !defined(GGML_USE_HIP)
// cuBLAS-GEMM flash attention for pre-Volta NVIDIA. On by default there; GGML_CUDA_FA_GEMM=0 disables.
// See fattn-gemm.cu for why this path exists.
bool ggml_cuda_fa_gemm_enabled();
bool ggml_cuda_flash_attn_ext_gemm_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif // defined(GGML_USE_HIP)
