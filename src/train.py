import argparse, json, os, random, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from model import IDSMTran, FocalLoss
from data import Prep, make_splits


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def evaluate(m, X, y, dev):
    m.eval(); P = []
    with torch.no_grad():
        for xb, in DataLoader(TensorDataset(X), batch_size=2048):
            P.append(m(xb.to(dev)).argmax(1).cpu())
    p = torch.cat(P).numpy(); y = y.numpy()
    return dict(acc=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro")),
                per_class_recall=recall_score(y, p, average=None, zero_division=0).round(4).tolist())


def fit(Xtr, ytr, Xva, yva, n_classes, grid, epochs=100, bs=512, lr=1e-3, patience=15, dev="cpu", loss="focal", **mk):
    m = IDSMTran(grid=grid, n_classes=n_classes, **mk).to(dev)
    cnt = np.bincount(ytr.numpy(), minlength=n_classes)
    alpha = torch.tensor((cnt.sum() / (n_classes * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32)
    crit = FocalLoss(alpha / alpha.mean(), 2.0) if loss == "focal" else torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.99, 0.9999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    best, bad, state = -1, 0, None
    for ep in range(epochs):
        m.train()
        for xb, yb in DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True):
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        r = evaluate(m, Xva, yva, dev); sched.step(r["macro_f1"])
        print(f"ep{ep:03d} val acc {r['acc']:.4f} macroF1 {r['macro_f1']:.4f}")
        if r["macro_f1"] > best:
            best, bad, state = r["macro_f1"], 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nsl", choices=["nsl", "unsw", "cic"])
    ap.add_argument("--train"); ap.add_argument("--test")
    ap.add_argument("--protocol", default="official", choices=["official", "pooled"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--name", default="idsmtran", help="variant name, used in result filenames")
    ap.add_argument("--branches", default="0,1,2"); ap.add_argument("--patches", default="2,3,4")
    ap.add_argument("--no_cfe", action="store_true"); ap.add_argument("--loss", default="focal", choices=["focal", "ce"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(a.seed)

    tr, va, te, cat = make_splits(a.dataset, a.train, a.test, a.protocol, a.seed)
    pp = Prep(cat).fit(tr)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = [tuple(map(torch.tensor, pp.transform(d))) for d in (tr, va, te)]
    mk = dict(branches=tuple(int(i) for i in a.branches.split(",")), patches=tuple(int(i) for i in a.patches.split(",")),
              use_cfe=not a.no_cfe)
    m = fit(Xtr, ytr, Xva, yva, len(pp.le.classes_), pp.grid, a.epochs, dev=dev, loss=a.loss, **mk)

    res = evaluate(m, Xte, yte, dev)
    res.update(classes=list(pp.le.classes_), dataset=a.dataset, protocol=a.protocol, seed=a.seed, model=a.name)
    print("TEST", res)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dataset}_{a.protocol}_seed{a.seed}" if a.name == "idsmtran" else f"{a.name}_{a.dataset}_{a.protocol}_seed{a.seed}"
    json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=2)
    torch.save(m.state_dict(), f"{a.out}/{tag}.pt")
