import pandas as pd
import argparse, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from model import IDSMTran, FocalLoss
from data import load_nsl, Prep


def evaluate(m, X, y, dev):
    m.eval(); P = []
    with torch.no_grad():
        for xb, in DataLoader(TensorDataset(X), batch_size=2048):
            P.append(m(xb.to(dev)).argmax(1).cpu())
    p = torch.cat(P).numpy(); y = y.numpy()
    return dict(acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"),
                per_class_recall=recall_score(y, p, average=None, zero_division=0).round(3).tolist())


def fit(Xtr, ytr, Xva, yva, n_classes, grid, epochs=100, bs=512, lr=1e-3, patience=10, dev="cpu"):
    m = IDSMTran(grid=grid, n_classes=n_classes).to(dev)
    cnt = np.bincount(ytr.numpy(), minlength=n_classes)
    alpha = torch.tensor((cnt.sum() / (n_classes * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32)  # sqrt inverse-freq
    crit = FocalLoss(alpha / alpha.mean(), 2.0)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.99, 0.9999))
    best, bad, state = -1, 0, None
    for ep in range(epochs):
        m.train()
        for xb, yb in DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True):
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        r = evaluate(m, Xva, yva, dev)
        print(f"ep{ep:03d} val acc {r['acc']:.4f} macroF1 {r['macro_f1']:.4f}")
        if r["macro_f1"] > best: best, bad, state = r["macro_f1"], 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--test")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--protocol", default="official", choices=["official", "pooled"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr0, te0, cat = load_nsl(a.train, a.test)
    if a.protocol == "pooled":
        full = pd.concat([tr0, te0]).drop_duplicates()
        tr, te = train_test_split(full, test_size=0.2, stratify=full["label"], random_state=0)
    else:
        tr, te = tr0, te0
    tr, va = train_test_split(tr, test_size=0.1, stratify=tr["label"], random_state=0)  # val carved BEFORE fitting prep

    pp = Prep(cat).fit(tr)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = [tuple(map(torch.tensor, pp.transform(d))) for d in (tr, va, te)]
    m = fit(Xtr, ytr, Xva, yva, len(pp.le.classes_), pp.grid, a.epochs, dev=dev)
    print("classes:", list(pp.le.classes_)); print("TEST", evaluate(m, Xte, yte, dev))