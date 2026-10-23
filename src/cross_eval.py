"""Train on one NF-v2 dataset, test on another (binary benign vs attack, shared features only)."""
import argparse, json, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from data import Prep

META = {"label", "binary"}
ap = argparse.ArgumentParser()
ap.add_argument("--train_parquet", required=True); ap.add_argument("--test_parquet", required=True)
ap.add_argument("--model", default="idsmtran", choices=["idsmtran", "rf"])
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--grid", type=int, default=8); ap.add_argument("--out", default="../results")
a = ap.parse_args()
stem = lambda p: os.path.basename(p).replace(".parquet", "").replace("nf_", "")
A, B = pd.read_parquet(a.train_parquet), pd.read_parquet(a.test_parquet)
feats = sorted((set(A.columns) & set(B.columns)) - META)
print(f"{stem(a.train_parquet)} -> {stem(a.test_parquet)}: {len(feats)} shared features")


def frame(df):
    d = df[feats].copy(); d["label"] = df["binary"].map({0: "benign", 1: "attack"}); return d


trd, ted = frame(A), frame(B)
tr, va = train_test_split(trd, test_size=0.1, stratify=trd["label"], random_state=a.seed)
pp = Prep([], grid=a.grid).fit(tr)
(Xtr, ytr), (Xva, yva), (Xte, yte) = [pp.transform(d) for d in (tr, va, ted)]

if a.model == "rf":
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=a.seed).fit(Xtr, ytr)
    p = rf.predict(Xte)
    res = dict(acc=float(accuracy_score(yte, p)), macro_f1=float(f1_score(yte, p, average="macro")),
               per_class_recall=recall_score(yte, p, average=None, zero_division=0).round(4).tolist())
else:
    import torch
    from train import fit, evaluate, set_seed
    set_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda X, y: (torch.tensor(X), torch.tensor(y))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = T(Xtr, ytr), T(Xva, yva), T(Xte, yte)
    m = fit(Xtr, ytr, Xva, yva, 2, pp.grid, a.epochs, dev=dev)
    res = evaluate(m, Xte, yte, dev)

cls = list(pp.le.classes_)                         # ['attack', 'benign']
res.update(classes=cls, attack_recall=res["per_class_recall"][cls.index("attack")],
           benign_recall=res["per_class_recall"][cls.index("benign")],
           dataset=f"{stem(a.train_parquet)}->{stem(a.test_parquet)}", protocol="cross", seed=a.seed, model=f"{a.model}_cross")
print("TEST", res)
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/cross_{a.model}_{stem(a.train_parquet)}_{stem(a.test_parquet)}_seed{a.seed}.json", "w"), indent=2)
