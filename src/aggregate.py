"""Mean +- std over seeds for every (model, dataset, protocol) found in results/*.json."""
import glob, json, sys, numpy as np
from collections import defaultdict
d = sys.argv[1] if len(sys.argv) > 1 else "../results"
g = defaultdict(list)
for f in glob.glob(f"{d}/*.json"):
    r = json.load(open(f)); g[(r["model"], r["dataset"], r["protocol"])].append(r)
print(f"{'model':16}{'data':12}{'protocol':10}{'n':>3}  {'acc':>14}  {'macroF1':>14}")
for k, rs in sorted(g.items()):
    a = np.array([r["acc"] for r in rs]) * 100; f = np.array([r["macro_f1"] for r in rs])
    print(f"{k[0]:16}{k[1]:12}{k[2]:10}{len(rs):>3}  {a.mean():6.2f} ± {a.std():4.2f}  {f.mean():6.3f} ± {f.std():5.3f}")
