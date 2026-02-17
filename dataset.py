import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS_PATH = "labels.csv"
TRAIN_OUT = "train.csv"
VAL_OUT = "val.csv"
VAL_RATIO = 0.2
SEED = 42

df = pd.read_csv(LABELS_PATH)

# Нормализуем пути под текущую ОС (если fullfile писал обратные слэши)
df["path"] = df["path"].astype(str).str.replace("\\", os.sep).str.replace("/", os.sep)

# Сплит по ref (чтобы не было утечки)
refs = df["ref"].astype(str).unique()
train_refs, val_refs = train_test_split(refs, test_size=VAL_RATIO, random_state=SEED, shuffle=True)

train_df = df[df["ref"].astype(str).isin(train_refs)].reset_index(drop=True)
val_df   = df[df["ref"].astype(str).isin(val_refs)].reset_index(drop=True)

train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)

print("Saved:", TRAIN_OUT, len(train_df))
print("Saved:", VAL_OUT, len(val_df))
print("Unique refs train/val:", len(train_refs), len(val_refs))
