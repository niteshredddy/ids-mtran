import torch, numpy as np
from model import IDSMTran, FocalLoss
from train import fit, evaluate
torch.manual_seed(0)
X = torch.rand(2000, 144); y = (X[:, :5].sum(1) * 1.2).long().clamp(0, 4)
m = fit(X[:1600], y[:1600], X[1600:], y[1600:], 5, 12, epochs=3)
print(evaluate(m, X[1600:], y[1600:], "cpu"), sum(p.numel() for p in m.parameters()), "params")
