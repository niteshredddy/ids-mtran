"""Turn the huge CIC-DDoS2019 CSVs into two small parquet files (train/test) with a uniform per-class row cap.
Streams in chunks (low RAM). Labels are normalised (DrDoS_DNS -> DNS, UDP-lag -> UDPLag, BENIGN -> Benign) so the
01-12 and 03-11 days use the same class names.
  python prepare_cic.py --peek file.csv
  python prepare_cic.py --train_csv a.csv b.csv --test_csv c.csv --out_dir DIR --cap 30000"""
import argparse, os, re, numpy as np, pandas as pd

DROP = {"unnamed: 0", "flow id", "source ip", "destination ip", "src ip", "dst ip", "timestamp", "simillarhttp", "similarhttp"}


def clean_cols(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def norm_label(s):
    s = str(s).strip()
    s = re.sub(r"^DrDoS_", "", s)
    low = s.lower()
    if low in ("udp-lag", "udplag"): return "UDPLag"
    if low == "benign": return "Benign"
    return s


def reservoir_update(store, lab, g, cap, rng):
    """Keep a uniform random sample of <= cap rows per label across all chunks (smallest random keys win)."""
    g = g.copy(); g["_k"] = rng.random(len(g))
    if lab in store: g = pd.concat([store[lab], g])
    store[lab] = g.nsmallest(cap, "_k")


def sample_files(files, cap, seed, chunksize=400_000):
    rng = np.random.default_rng(seed); store = {}
    for f in files:
        print("reading", os.path.basename(f), flush=True)
        for ch in pd.read_csv(f, chunksize=chunksize, low_memory=False):
            ch = clean_cols(ch)
            ch = ch.drop(columns=[c for c in ch.columns if c.lower() in DROP])
            ch = ch.rename(columns={"Label": "label"})
            ch["label"] = ch["label"].map(norm_label)
            feats = [c for c in ch.columns if c != "label"]
            ch[feats] = ch[feats].apply(pd.to_numeric, errors="coerce")
            ch = ch.replace([np.inf, -np.inf], np.nan).dropna()
            for lab, g in ch.groupby("label"):
                reservoir_update(store, lab, g, cap, rng)
    df = pd.concat(store.values()).drop(columns="_k").reset_index(drop=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek"); ap.add_argument("--train_csv", nargs="*"); ap.add_argument("--test_csv", nargs="*")
    ap.add_argument("--out_dir"); ap.add_argument("--cap", type=int, default=30000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.peek:
        df = clean_cols(pd.read_csv(a.peek, nrows=200_000, low_memory=False))
        print(len(df.columns), "columns:", list(df.columns)); print(df["Label"].map(norm_label).value_counts()); raise SystemExit
    os.makedirs(a.out_dir, exist_ok=True)
    for name, files in (("train", a.train_csv), ("test", a.test_csv)):
        df = sample_files(files, a.cap, a.seed)
        print(name, df.shape); print(df["label"].value_counts())
        df.to_parquet(f"{a.out_dir}/cic_{name}.parquet")
