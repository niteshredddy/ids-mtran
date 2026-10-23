"""Sample an NF-v2 NetFlow dataset (zip or csv) into a small parquet. Streams in chunks.
Output columns: the numeric flow features, 'label' (attack name) and 'binary' (0 benign / 1 attack).
  python prepare_nf.py --zip NF-x.zip --peek
  python prepare_nf.py --zip NF-x.zip --out nf_x.parquet --cap 30000"""
import argparse, io, zipfile, numpy as np, pandas as pd
from prepare_cic import reservoir_update

DROP = {"ipv4_src_addr", "ipv4_dst_addr", "l4_src_port", "l4_dst_port", "flow_start_milliseconds", "flow_end_milliseconds"}


def open_csv(path, member=None):
    if path.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path)
        csvs = [i for i in zf.infolist() if i.filename.lower().endswith(".csv") and "feature" not in i.filename.lower()]
        info = max(csvs, key=lambda i: i.file_size) if member is None else zf.getinfo(member)
        print("using", info.filename, f"({info.file_size/1e6:.0f} MB)")
        return zf.open(info)
    return open(path, "rb")


def find(cols, name):
    for c in cols:
        if c.strip().lower() == name: return c
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True); ap.add_argument("--member"); ap.add_argument("--out")
    ap.add_argument("--cap", type=int, default=30000, help="max rows per attack class")
    ap.add_argument("--benign_cap", type=int, default=0, help="max benign rows (default 5*cap)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--peek", action="store_true")
    a = ap.parse_args()
    if a.peek:
        df = pd.read_csv(open_csv(a.zip, a.member), nrows=50_000)
        print(len(df.columns), "columns:", list(df.columns))
        for c in ("Attack", "Label"):
            if find(df.columns, c.lower()): print(df[find(df.columns, c.lower())].value_counts().head(15))
        raise SystemExit
    rng = np.random.default_rng(a.seed); store = {}; bcap = a.benign_cap or 5 * a.cap
    for ch in pd.read_csv(open_csv(a.zip, a.member), chunksize=400_000, low_memory=False):
        att, lab = find(ch.columns, "attack"), find(ch.columns, "label")
        name = ch[att].astype(str).str.strip() if att else ch[lab].map({0: "Benign", 1: "Attack"}).astype(str)
        ch = ch.drop(columns=[c for c in ch.columns if c.strip().lower() in DROP or c in (att, lab)])
        ch = ch.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        ch["label"] = name.values
        ch = ch.dropna()
        for l, g in ch.groupby("label"):
            reservoir_update(store, l, g, bcap if l.lower() == "benign" else a.cap, rng)
    df = pd.concat(store.values()).drop(columns="_k").reset_index(drop=True)
    df["binary"] = (df["label"].str.lower() != "benign").astype(int)
    print(df.shape); print(df["label"].value_counts()); print("binary:", df["binary"].value_counts().to_dict())
    df.to_parquet(a.out)
