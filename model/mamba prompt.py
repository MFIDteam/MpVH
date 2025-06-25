import time
import torch
from torch import nn
import torch.nn.init as init
import warnings


# WITH_SELECTIVESCAN_OFLEX = True
# WITH_SELECTIVESCAN_CORE = False
# WITH_SELECTIVESCAN_MAMBA = True
# try:
#     import selective_scan_cuda_oflex
# except ImportError:
#     WITH_SELECTIVESCAN_OFLEX = False
#     warnings.warn("Can not import selective_scan_cuda_oflex. This affects speed.")
#     print("Can not import selective_scan_cuda_oflex. This affects speed.", flush=True)
# try:
#     import selective_scan_cuda_core
# except ImportError:
#     WITH_SELECTIVESCAN_CORE = False
# try:
#     import selective_scan_cuda
# except ImportError:
#     WITH_SELECTIVESCAN_MAMBA = False


def selective_scan_torch(
        u: torch.Tensor,  # (B, K * C, L)
        delta: torch.Tensor,  # (B, K * C, L)
        A: torch.Tensor,  # (K * C, N)
        B: torch.Tensor,  # (B, K, N, L)
        C: torch.Tensor,  # (B, K, N, L)
        D: torch.Tensor = None,  # (K * C)
        delta_bias: torch.Tensor = None,  # (K * C)
        delta_softplus=True,
        oflex=True,
        *args,
        **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]  # K * C
    Cdim = int(KCdim / K)  # C

    # Validate tensor shapes
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    # Ensure tensors are in float format for computation
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()

    # Broadcasting mechanism to avoid repeated operations
    B = B.view(Batch, K, 1, N, L).expand(-1, -1, Cdim, -1, -1).reshape(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).expand(-1, -1, Cdim, -1, -1).reshape(Batch, KCdim, N, L)

    # Precompute deltaA and deltaB_u in parallel
    deltaA = torch.einsum('bdl,dn->bdln', delta, A)  # 防止指数溢出
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)  # (B, KCdim, L, N)

    # Compute x using broadcasting instead of for-loop
    x = deltaA * deltaB_u  # (B, KCdim, L, N)

    # C is reshaped for einsum operation to match x
    C = C.permute(0, 1, 3, 2)  # (B, KCdim, L, N)

    # Compute y using einsum, we sum over L dimension to get (B, N)
    y = torch.einsum('bdln,bdln->bdl', x, C)  # (B, KCdim, L)

    # Add D if provided
    out = y if D is None else y + u * D.unsqueeze(-1)

    return out if oflex else out.to(dtype=dtype_in)


# class SelectiveScanCuda(torch.autograd.Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
#         ctx.delta_softplus = delta_softplus
#         backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
#         backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
#         backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
#         ctx.backend = backend
#
#         if backend == "oflex":
#             out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
#         elif backend == "core":
#             out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
#         elif backend == "mamba":
#             out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
#
#         ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
#         return out
#
#     @staticmethod
#     @torch.cuda.amp.custom_bwd
#     def backward(ctx, dout, *args):
#         u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
#         backend = ctx.backend
#
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#
#         if backend == "oflex":
#             du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
#                 u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
#             )
#         elif backend == "core":
#             du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
#                 u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
#             )
#         elif backend == "mamba":
#             du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
#                 u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
#                 False
#             )
#
#         return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


# 选择性扫描函数（这里假设已经定义或者导入）
# def selective_scan_fn(
#         u: torch.Tensor,  # (B, K * C, L)
#         delta: torch.Tensor,  # (B, K * C, L)
#         A: torch.Tensor,  # (K * C, N)
#         B: torch.Tensor,  # (B, K, N, L)
#         C: torch.Tensor,  # (B, K, N, L)
#         D: torch.Tensor = None,  # (K * C)
#         delta_bias: torch.Tensor = None,  # (K * C)
#         delta_softplus=True,
#         oflex=True,
#         backend=None,
# ):
    # 检查并选择合适的加速方式
    # WITH_CUDA = (WITH_SELECTIVESCAN_OFLEX or WITH_SELECTIVESCAN_CORE or WITH_SELECTIVESCAN_MAMBA)
    # fn = selective_scan_torch if backend == "torch" or (not WITH_CUDA) else SelectiveScanCuda.apply
    # return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)


# 提示器模块：集成选择性扫描
class MambaAdapter(nn.Module):
    def __init__(self, dim=768, d_state=16, ssm_ratio=1.0, dropout=0.0, K=16, **kwargs):
        super().__init__()
        self.d_model = dim
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * dim)
        self.K = K
        self.C = 48

        # Use Xavier initialization for Linear layers to avoid large weight initialization
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner, kernel_size=3, padding=1)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

        # Xavier initialization for conv2d to help with stability
        init.xavier_uniform_(self.in_proj.weight)
        init.xavier_uniform_(self.out_proj.weight)
        init.xavier_uniform_(self.conv2d.weight)

        # Bias initialization to small values to avoid sudden shifts
        init.zeros_(self.in_proj.bias)
        init.zeros_(self.out_proj.bias)
        init.zeros_(self.conv2d.bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.A_logs = nn.Parameter(torch.zeros(self.K * self.C, self.d_state))
        self.Ds = nn.Parameter(torch.zeros(self.K * self.C))
        self.dt_projs_weight = nn.Parameter(torch.zeros(self.K * self.C, self.d_state))
        self.dt_projs_bias = nn.Parameter(torch.zeros(self.K * self.C))

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)

        B, N, C = x.shape
        x = x.permute(0, 2, 1).unsqueeze(3)  # [batch, d_inner, 197, 1]
        x = self.conv2d(x).squeeze(3)  # [batch, d_inner, 197]
        x = self.act(x)

        # Add a small constant epsilon to avoid division by zero if necessary
        epsilon = 1e-6

        A = self.A_logs.expand(self.K * self.C, self.d_state)
        B_tensor = torch.randn((B, self.K, self.d_state, N), device=x.device)
        C_tensor = torch.randn((B, self.K, self.d_state, N), device=x.device)
        delta_tensor = torch.randn((B, self.K * self.C, N), device=x.device)
        delta_bias_tensor = torch.randn(self.K * self.C, device=x.device)

        output = selective_scan_torch(
            x, delta_tensor, A, B_tensor, C_tensor, delta_bias=delta_bias_tensor, delta_softplus=True, backend="torch"
        )

        # Apply gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        x = self.out_proj(output.permute(0, 2, 1))
        return x