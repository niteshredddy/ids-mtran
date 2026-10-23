"""IDS-MTran in PyTorch: 3 scale branches -> PwP -> Transformer x2 each -> CFE -> 3 linear layers.
The paper leaves grid size, patch sizes and CFE details open; choices here are marked (ASSUMPTION)."""
import math
import torch, torch.nn as nn, torch.nn.functional as F


class PwP(nn.Module):
    """Patching with Pooling: avg-pool (denoise) -> upsample (restore detail) -> residual, then patchify."""
    def __init__(self, ch, d, patch, pool=2):
        super().__init__()
        self.pool = pool
        self.mix = nn.Conv2d(ch, ch, 1)
        self.embed = nn.Conv2d(ch, d, patch, stride=patch)  # patch embedding (T = (h/s)*(w/s) tokens)

    def forward(self, x):
        p = F.avg_pool2d(x, self.pool, ceil_mode=True)
        u = F.interpolate(p, size=x.shape[-2:], mode="nearest")
        x = x + self.mix(u)
        t = self.embed(x)                                   # B,d,gh,gw
        return t


def sincos(n, d, device):
    pos = torch.arange(n, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


def cbg(i, o, dil=1):
    k = 1 if dil == 0 else 3
    d = max(dil, 1)
    return nn.Sequential(nn.Conv2d(i, o, k, padding=0 if k == 1 else d, dilation=d), nn.BatchNorm2d(o), nn.GELU())


class Backbone(nn.Module):
    def __init__(self, d, heads, layers, drop):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, t):                                   # t: B,d,gh,gw
        B, d, gh, gw = t.shape
        x = t.flatten(2).transpose(1, 2)
        x = x + sincos(x.size(1), d, x.device)
        x = self.enc(x)
        return x.transpose(1, 2).reshape(B, d, gh, gw)


class CFE(nn.Module):
    """Cross Feature Enrichment: resize every scale into every branch, concat, 1x1 conv, down-sample, fuse, 3 linear layers."""
    def __init__(self, d, n_classes, drop, n=3):
        super().__init__()
        self.mix = nn.ModuleList([nn.Conv2d(n * d, d, 1) for _ in range(n)])
        self.head = nn.Sequential(nn.Linear(n * d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_classes))

    def forward(self, feats):
        enriched = []
        for i, f in enumerate(feats):
            size = f.shape[-2:]
            cat = torch.cat([F.interpolate(g, size=size, mode="bilinear", align_corners=False) if g.shape[-2:] != size else g
                             for g in feats], 1)
            e = self.mix[i](cat)
            enriched.append(F.adaptive_avg_pool2d(e, 1).flatten(1))   # distil
        return self.head(torch.cat(enriched, 1))


class IDSMTran(nn.Module):
    """branches: subset of (0,1,2) = P1,P2,P3. use_cfe=False -> pooled concat + 3 linear layers (paper's ablation)."""
    def __init__(self, grid=12, n_classes=5, ch=16, d=64, heads=4, layers=2,
                 patches=(2, 3, 4), drop=0.1, branches=(0, 1, 2), use_cfe=True):
        super().__init__()
        self.grid, self.branches = grid, tuple(branches)
        self.full = self.branches == (0, 1, 2)               # top-down P1->P2->P3 fusion only in the full model
        self.use_cfe = use_cfe and len(self.branches) > 1
        mk = {0: lambda: cbg(1, ch, 0),
              1: lambda: nn.Sequential(cbg(1, ch, 0), cbg(ch, ch, 2)),
              2: lambda: nn.Sequential(cbg(1, ch, 0), cbg(ch, ch, 2), cbg(ch, ch, 2))}
        k = [str(i) for i in self.branches]
        self.convs = nn.ModuleDict({i: mk[int(i)]() for i in k})
        self.pwp = nn.ModuleDict({i: PwP(ch, d, patches[int(i)]) for i in k})
        self.bb = nn.ModuleDict({i: Backbone(d, heads, layers, drop) for i in k})
        n = len(k)
        if self.use_cfe:
            self.cfe = CFE(d, n_classes, drop, n)
        else:
            self.head = nn.Sequential(nn.Linear(n * d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                      nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_classes))

    def forward(self, x):                                   # x: B, grid*grid (padded features)
        x = x.view(-1, 1, self.grid, self.grid)
        f = {i: self.convs[str(i)](x) for i in self.branches}
        if self.full:                                       # low-level (auxiliary) supplements high-level (primary)
            f[1] = f[1] + f[0]; f[2] = f[2] + f[1]
        feats = [self.bb[str(i)](self.pwp[str(i)](f[i])) for i in self.branches]
        if self.use_cfe:
            return self.cfe(feats)
        return self.head(torch.cat([F.adaptive_avg_pool2d(t, 1).flatten(1) for t in feats], 1))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
        pt = logp.exp()
        loss = -((1 - pt) ** self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[y]
        return loss.mean()
