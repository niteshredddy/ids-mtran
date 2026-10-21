"""Random Forest baseline on the exact same splits/preprocessing as train.py."""
import argparse, json, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from data import Prep, make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nsl"); ap.add_argument("--train"); ap.add_argument("--test")
ap.add_argument("--protocol", default="official"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="../results")
a = ap.parse_args()

tr, va, te, cat = make_splits(a.dataset, a.train, a.test, a.protocol, a.seed)
pp = Prep(cat).fit(tr)
Xtr, ytr = pp.transform(tr); Xte, yte = pp.transform(te)
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=a.seed).fit(Xtr, ytr)
p = rf.predict(Xte)
res = dict(acc=float(accuracy_score(yte, p)), macro_f1=float(f1_score(yte, p, average="macro")),
           per_class_recall=recall_score(yte, p, average=None, zero_division=0).round(4).tolist(),
           classes=list(pp.le.classes_), dataset=a.dataset, protocol=a.protocol, seed=a.seed, model="rf")
print("TEST", res)
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/rf_{a.dataset}_{a.protocol}_seed{a.seed}.json", "w"), indent=2)
