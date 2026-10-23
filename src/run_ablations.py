"""Runs every ablation variant x seed on NSL-KDD official split. Skips finished runs, so it is safe to re-run after a disconnect."""
import argparse, os, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--dataset", default="nsl")
ap.add_argument("--protocol", default="official"); ap.add_argument("--seeds", default="0,1")
ap.add_argument("--epochs", default="40")
a = ap.parse_args()
VARIANTS = {
    "idsmtran": [],
    "abl_noCFE": ["--no_cfe"],
    "abl_P1only": ["--branches", "0"],
    "abl_P2only": ["--branches", "1"],
    "abl_P3only": ["--branches", "2"],
    "abl_CE": ["--loss", "ce"],
    "abl_patch123": ["--patches", "1,2,3"],
}
for name, extra in VARIANTS.items():
    for seed in a.seeds.split(","):
        tag = f"{a.dataset}_{a.protocol}_seed{seed}" if name == "idsmtran" else f"{name}_{a.dataset}_{a.protocol}_seed{seed}"
        if os.path.exists(f"{a.out}/{tag}.json"):
            print("skip", tag); continue
        cmd = [sys.executable, "train.py", "--dataset", a.dataset, "--train", a.train, "--test", a.test,
               "--protocol", a.protocol, "--seed", seed, "--epochs", a.epochs, "--out", a.out, "--name", name] + extra
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)
