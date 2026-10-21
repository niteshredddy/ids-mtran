"""Leakage-safe preprocessing: dedupe -> split -> fit encoder/scaler/clip on TRAIN only -> pad to h*w grid."""
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

NSL_COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
 "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
 "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
 "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
 "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
 "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
DOS = "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb".split()
PROBE = "satan ipsweep nmap portsweep mscan saint".split()
R2L = "guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named".split()
U2R = "buffer_overflow loadmodule rootkit perl sqlattack xterm ps".split()


def nsl_group(a):
    a = a.strip().lower()
    return "Normal" if a == "normal" else "DoS" if a in DOS else "Probe" if a in PROBE else "R2L" if a in R2L else "U2R"


def load_nsl(train_path, test_path):
    tr = pd.read_csv(train_path, names=NSL_COLS).drop(columns="difficulty")
    te = pd.read_csv(test_path, names=NSL_COLS).drop(columns="difficulty")
    for df in (tr, te):
        df["label"] = df["label"].map(nsl_group)
    tr = tr.drop_duplicates()
    return tr, te, ["protocol_type", "service", "flag"]


class Prep:
    def __init__(self, cat_cols, grid=None, clip_q=(0.001, 0.999)):
        self.cat, self.grid, self.q = cat_cols, grid, clip_q

    def fit(self, df, label="label"):
        self.num = [c for c in df.columns if c not in self.cat + [label]]
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[self.cat]) if self.cat else None
        self.lo, self.hi = df[self.num].quantile(self.q[0]), df[self.num].quantile(self.q[1])   # clip, not 3-sigma deletion
        self.sc = MinMaxScaler().fit(df[self.num].clip(self.lo, self.hi, axis=1))                # train-only scaler
        self.le = LabelEncoder().fit(df[label])
        n = len(self.num) + (self.ohe.transform(df[self.cat].head(1)).shape[1] if self.ohe else 0)
        self.grid = self.grid or int(np.ceil(np.sqrt(n)))
        self.label = label
        return self

    def transform(self, df):
        num = self.sc.transform(df[self.num].clip(self.lo, self.hi, axis=1))
        X = np.hstack([num, self.ohe.transform(df[self.cat])]) if self.ohe else num
        X = np.clip(X, 0, 1).astype(np.float32)
        pad = self.grid ** 2 - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad)))                       # zero-fill the tail, as in the paper
        return X, self.le.transform(df[self.label]).astype(np.int64)
