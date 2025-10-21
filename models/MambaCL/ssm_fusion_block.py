from models.MambaCL.vmamba import SelectiveScanOflex, LayerNorm2d
import torch.nn as nn
import torch
from einops import rearrange, repeat
import torch.nn.functional as F
import math


class SSM_fusion(nn.Module):
    def __init__(
            self,
            d_model=96,
            ssm_ratio=2.0,
            d_state=16,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
    ):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}
        self.k_groups = 2
  
        d_inner = 32
        dt_rank = 32
        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        self.selective_scan = SelectiveScanOflex
        self.A_logs = self.A_log_init(d_state, d_inner, copies=self.k_groups, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.k_groups, merge=True)  # (K * D)
        self.out_norm = LayerNorm2d(d_inner)
        layers = []
        layers.append(nn.Conv2d(768, 32, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(416, 32, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(224, 32, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(128, 32, kernel_size=3, padding=1))
        self.convs = nn.Sequential(*layers)


    def forward(self, outs):
        fusion_x = outs[-1]
        fusion_x = self.convs[0](fusion_x)
        for i in range(len(outs)-1):
            B, C, H, W = outs[2-i].size()
            x = F.interpolate(fusion_x, size=(H,W), mode='bicubic', align_corners=True)
            x = torch.cat([x, outs[2-i]],dim=1)
            x = self.convs[i+1](x)
            D, N = self.A_logs.shape
            K, D, R = self.dt_projs_weight.shape
            L = H*W

            xs = x.new_empty((B, self.k_groups, 32, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(self.A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = self.Ds.to(torch.float)  # (K * c)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)
            ys = self.selective_scan.apply(xs, dts, As, Bs, Cs, Ds, delta_bias, True).view(B, K, D, -1)
            fusion_x = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            fusion_x = fusion_x.view(B,-1,H,W)
            fusion_x = self.out_norm(fusion_x)

        return fusion_x

    def dt_init(self, dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    def A_log_init(self, d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    def D_init(self, d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D