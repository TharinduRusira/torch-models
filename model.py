import torch
from torch.nn.functional import scaled_dot_product_attention, multi_head_attention_forward

import torch._inductor.config as config

config.trace.enabled = True
config.cpp_wrapper = False
config.trace.debug_dir = './.torch_debug'
config.benchmark_kernel = True
# config.max_autotune = True
# config.max_autotune_gemm = True
config.max_autotune_gemm_backends = "CPP"
config.cpp.vec_isa_ok = True
config.comment_origin = True
config.autotune_fallback_to_aten = False


class DefaultAttention(torch.nn.Module):
    def __init__(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.q = q
        self.v = v
        super().__init__()
        self.training= False

    def forward(self):
        return scaled_dot_product_attention(self.q, self.k, self.v, attn_mask=None, dropout_p=0.0)

class CustomAttention(torch.nn.Module):
    def __init__(self, k, q, v):
        self.k = k
        self.q = q
        self.v = v

    def forward(self, x):
        pass

class Transformer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        pass

class MatMul(torch.nn.Module):
    def forward(self, x, y):
        c = torch.zeros((x.shape[0], y.shape[1]), dtype=x.dtype)
        for i in range(x.shape[0]):
            for j in range(y.shape[1]):
                for k in range(x.shape[1]):
                    c[i, j] += x[i, k] * y[k, j]
        return c


def run_model_inference():
    k = torch.rand((16, 16), dtype=torch.float16)
    q = torch.rand((16, 16), dtype=torch.float16)
    v = torch.rand((16, 16), dtype=torch.float16)
    default_attn = DefaultAttention(k, q, v)
    compiled_model = torch.compile(default_attn, backend="inductor")
    return compiled_model()

if __name__ == '__main__':
    run_model_inference()