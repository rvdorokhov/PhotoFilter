import pandas as pd
import numpy as np

LABELS_CSV = "night_labels.csv"
TRAIN_OUT = "night_train.csv"
VAL_OUT = "night_val.csv"

VAL_RATIO = 0.20
SEED = 42

df = pd.read_csv(LABELS_CSV)

# пути под Linux
df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
df["path"] = df["path"].str.replace("//", "/", regex=False)
df["ref"]  = df["ref"].astype(str).str.replace("\\", "/", regex=False)

rng = np.random.RandomState(SEED)

# stratified split по night на уровне ref
ref_night = df.groupby("ref")["night"].max().reset_index()
refs_1 = ref_night.loc[ref_night["night"] == 1, "ref"].values
refs_0 = ref_night.loc[ref_night["night"] == 0, "ref"].values

rng.shuffle(refs_1)
rng.shuffle(refs_0)

n_val_1 = int(round(len(refs_1) * VAL_RATIO))
n_val_0 = int(round(len(refs_0) * VAL_RATIO))

val_refs = set(refs_1[:n_val_1]) | set(refs_0[:n_val_0])
train_refs = set(refs_1[n_val_1:]) | set(refs_0[n_val_0:])

train_df = df[df["ref"].isin(train_refs)].reset_index(drop=True)
val_df   = df[df["ref"].isin(val_refs)].reset_index(drop=True)

train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)

print("Total rows:", len(df))
print("Train rows:", len(train_df), " Val rows:", len(val_df))
print("Night share total:", df["night"].mean())
print("Night share train:", train_df["night"].mean(), " val:", val_df["night"].mean())
print("Example path (train):", train_df.loc[0, "path"])
