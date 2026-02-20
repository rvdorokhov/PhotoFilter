# make_train_val_posix_small.py
import pandas as pd
import numpy as np

LABELS_CSV = "night_labels.csv"

SMALL_LABELS_OUT = "night_labels_test.csv"
TRAIN_OUT = "night_train_test.csv"
VAL_OUT = "night_val_test.csv"

VAL_RATIO = 0.20
SEED = 42

# сколько строк хотим в мини-версии (держи в диапазоне 300..500)
TARGET_N = 400
MIN_N = 300
MAX_N = 500

df = pd.read_csv(LABELS_CSV)

# 1) Нормализация путей
df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
df["path"] = df["path"].str.replace("//", "/", regex=False)

# ref тоже нормализуем (на всякий случай)
df["ref"] = df["ref"].astype(str).str.replace("\\", "/", regex=False)
df["ref"] = df["ref"].str.replace("//", "/", regex=False)

rng = np.random.RandomState(SEED)

def infer_group(p: str) -> str:
    """
    Пытаемся понять "категорию" по пути:
    - если путь содержит /Images/<категория>/... -> group = Images/<категория>
    - если путь содержит /ExDark/<категория>/... -> group = ExDark/<категория>
    иначе -> берём имя родительской папки
    """
    parts = [x for x in p.split("/") if x]
    for root in ("Images", "ExDark"):
        if root in parts:
            i = parts.index(root)
            if i + 1 < len(parts):
                return f"{root}/{parts[i+1]}"
    # fallback: родительская папка
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"

# 2) Группа (нужна, чтобы взять минимум 1 файл из каждой папки-категории)
df["group"] = df["path"].map(infer_group)

groups = df["group"].unique()
n_groups = len(groups)

# 3) Подбираем размер мини-датасета:
#    - минимум 1 на группу
#    - обычно 300..500
#    - если групп больше MAX_N, то сделать 300..500 невозможно при условии "из каждой папки"
if n_groups > MAX_N:
    print(f"[WARN] Групп (папок) = {n_groups} > {MAX_N}. "
          f"Нельзя уложиться в 300..500, сохраняя 'по 1 из каждой папки'. "
          f"Мини-датасет будет размером {n_groups}.")
    target_n = n_groups
else:
    target_n = max(n_groups, TARGET_N)
    target_n = max(MIN_N, min(MAX_N, target_n))

# 4) Гарантированно берём по 1 изображению из каждой группы
picked_idx = []
for g in groups:
    idxs = df.index[df["group"] == g].to_numpy()
    picked_idx.append(rng.choice(idxs))

picked_idx = set(picked_idx)

# 5) Добираем до target_n случайными строками
need_more = target_n - len(picked_idx)
if need_more > 0:
    remaining = np.array(list(set(df.index) - picked_idx))
    if need_more > len(remaining):
        need_more = len(remaining)
    extra = rng.choice(remaining, size=need_more, replace=False)
    picked_idx.update(extra.tolist())

df_small = df.loc[list(picked_idx)].copy()
df_small = df_small.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # перемешаем

# проверка: все группы присутствуют
missing_groups = set(groups) - set(df_small["group"].unique())
if missing_groups:
    print("[WARN] В мини-датасете пропали группы:", sorted(list(missing_groups))[:10], "...")
else:
    print(f"[OK] В мини-датасете есть хотя бы 1 изображение из каждой группы ({n_groups} групп).")

# сохраняем мини labels (удобно для контроля)
df_small.to_csv(SMALL_LABELS_OUT, index=False)

# 6) Разбиение train/val со стратификацией по night на уровне ref
# (у тебя ref почти уникален, но так разбиение будет стабильным и по доле night)
ref_night = df_small.groupby("ref")["night"].max().reset_index()

refs_1 = ref_night.loc[ref_night["night"] == 1, "ref"].values
refs_0 = ref_night.loc[ref_night["night"] == 0, "ref"].values

rng.shuffle(refs_1)
rng.shuffle(refs_0)

n_val_1 = int(round(len(refs_1) * VAL_RATIO))
n_val_0 = int(round(len(refs_0) * VAL_RATIO))

val_refs = set(refs_1[:n_val_1]) | set(refs_0[:n_val_0])
train_refs = set(refs_1[n_val_1:]) | set(refs_0[n_val_0:])

train_df = df_small[df_small["ref"].isin(train_refs)].reset_index(drop=True)
val_df   = df_small[df_small["ref"].isin(val_refs)].reset_index(drop=True)

train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)

print("=== MINI DATASET STATS ===")
print("Total rows (small):", len(df_small))
print("Train rows:", len(train_df), " Val rows:", len(val_df))
print("Groups:", n_groups)
print("Night share total:", df_small["night"].mean())
print("Night share train:", train_df["night"].mean(), " val:", val_df["night"].mean())
print("Example path (train):", train_df.loc[0, "path"])
print("Saved:", SMALL_LABELS_OUT, TRAIN_OUT, VAL_OUT)
