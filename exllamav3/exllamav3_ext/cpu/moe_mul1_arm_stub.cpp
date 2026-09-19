// aarch64 (and other non-x86) stand-in for moe_mul1.cpp, whose CPU MoE expert GEMM kernels are
// x86-only (AVX2 / AVX-512 intrinsics). Implements the public API from moe_mul1.h so the
// extension links; the compute entry points fail with a clear error. GPU inference is
// unaffected — only CPU MoE expert offload (and the swizzled-layout gates) are unavailable.
#include "moe_mul1.h"
#include <torch/extension.h>

#define NO_CPU_MOE "CPU MoE expert offload requires x86 (AVX2) and is not available on this platform"

int64_t exl3_moe_cpu_make_layer
(
    const std::vector<at::Tensor>& gate_trellis,
    const std::vector<at::Tensor>& gate_suh,
    const std::vector<at::Tensor>& gate_svh,
    const std::vector<at::Tensor>& up_trellis,
    const std::vector<at::Tensor>& up_suh,
    const std::vector<at::Tensor>& up_svh,
    const std::vector<at::Tensor>& down_trellis,
    const std::vector<at::Tensor>& down_suh,
    const std::vector<at::Tensor>& down_svh,
    const std::vector<at::Tensor>& gate_bias,
    const std::vector<at::Tensor>& up_bias,
    const std::vector<at::Tensor>& down_bias,
    int64_t activation,
    double act_limit,
    int64_t swizzled
)
{
    TORCH_CHECK(false, NO_CPU_MOE);
    return -1;
}

void exl3_moe_cpu_free_layer(int64_t handle) {}

void exl3_moe_cpu_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
)
{
    TORCH_CHECK(false, NO_CPU_MOE);
}

void exl3_moe_cpu_forward_raw
(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* w,
    float* out,
    int rows,
    int topk,
    int threads
)
{
    TORCH_CHECK(false, NO_CPU_MOE);
}

void exl3_moe_cpu_stage_experts
(
    int64_t handle,
    const uint32_t* expert_ids,
    int count,
    uint8_t* dst,
    int threads
)
{
    TORCH_CHECK(false, NO_CPU_MOE);
}

void exl3_moe_cpu_set_prof(bool enabled) {}
int64_t exl3_moe_cpu_pool_stress(int threads, int iters, int small, int spin) { return 0; }

bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_bw() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }
