# make_train_val_posix.py
import pandas as pd
import numpy as np

LABELS_CSV = "labels.csv"
TRAIN_OUT = "train.csv"
VAL_OUT = "val.csv"

VAL_RATIO = 0.20
SEED = 42

df = pd.read_csv(LABELS_CSV)

# 1) Делаем пути "линуксовскими": backslash -> slash
df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)

# (не обязательно, но полезно) уберём двойные слэши, если вдруг появились
df["path"] = df["path"].str.replace("//", "/", regex=False)

# 2) Split строго по ref (иначе в train и val попадут разные версии одной сцены)
refs = df["ref"].astype(str).unique()

rng = np.random.RandomState(SEED)
rng.shuffle(refs)

n_val = int(round(len(refs) * VAL_RATIO))
val_refs = set(refs[:n_val])
train_refs = set(refs[n_val:])

train_df = df[df["ref"].astype(str).isin(train_refs)].reset_index(drop=True)
val_df   = df[df["ref"].astype(str).isin(val_refs)].reset_index(drop=True)

train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)

print("Total rows:", len(df))
print("Train rows:", len(train_df), " Val rows:", len(val_df))
print("Unique refs train:", len(train_refs), " val:", len(val_refs))
print("Example path (train):", train_df.loc[0, "path"])
