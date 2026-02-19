# eval_val_metrics.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# -----------------
# Настройки
# -----------------
IMG_H, IMG_W = 384, 512

MODEL_PATH = "iq_multihd_savedmodel_1,5.keras"
TRAIN_CSV  = "train.csv"
VAL_CSV    = "val.csv"

# Пороги можно задавать разные для разных голов
TH_BLUR  = 0.50
TH_UNDER = 0.50
TH_OVER  = 0.50

# Для GTX1650 чаще всего безопасно 1..4
BATCH = 4

AUTOTUNE = tf.data.AUTOTUNE


# -----------------
# Data
# -----------------
def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)  # [0..255]
    img = tf.image.resize(img, (IMG_H, IMG_W), method="bilinear")
    return img

def make_ds_from_df(df, batch=BATCH):
    paths = df["path"].astype(str).tolist()
    labels = df[["blur", "under", "over"]].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, y):
        img = decode_and_resize(path)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds


# -----------------
# Метрики
# -----------------
def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0

def compute_metrics_from_counts(tp, fp, fn, tn):
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    acc       = safe_div(tp + tn, tp + tn + fp + fn)
    return precision, recall, f1, acc

def main():
    # GPU memory growth (чтобы не вылетало на некоторых конфигурациях)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # --- Читаем CSV ---
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    # --- Проверка утечки по ref ---
    if "ref" in train_df.columns and "ref" in val_df.columns:
        train_refs = set(train_df["ref"].astype(str).tolist())
        val_refs   = set(val_df["ref"].astype(str).tolist())
        overlap = train_refs.intersection(val_refs)

        print("\n[LEAK CHECK]")
        print("train refs:", len(train_refs))
        print("val refs:  ", len(val_refs))
        print("overlap refs (train ∩ val):", len(overlap))

        # Покажем пару примеров, если есть пересечение
        if len(overlap) > 0:
            sample = list(overlap)[:10]
            print("examples overlap refs:", sample)
            print("⚠️ Если overlap > 0, метрики на val могут быть завышены, потому что сцена уже видена моделью в train.")
    else:
        print("\n[LEAK CHECK] column 'ref' not found in one of CSVs, skip.")

    # --- Датасет val ---
    val_ds = make_ds_from_df(val_df, batch=BATCH)

    # --- Загружаем модель ---
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("\nLoaded model:", MODEL_PATH)

    # --- Счётчики по головам ---
    # Индексы: 0=blur, 1=under, 2=over
    counts = {
        "blur":  {"tp":0, "fp":0, "fn":0, "tn":0},
        "under": {"tp":0, "fp":0, "fn":0, "tn":0},
        "over":  {"tp":0, "fp":0, "fn":0, "tn":0},
    }

    total_samples = 0
    exact_match_correct = 0  # все 3 метки совпали
    micro_correct = 0        # правильных решений среди 3*N (TP+TN суммарно)
    micro_total = 0

    thresholds = np.array([TH_BLUR, TH_UNDER, TH_OVER], dtype=np.float32)

    # --- Прогон ---
    for x, y_true in val_ds:
        logits = model(x, training=False)         # (B,3) логиты
        y_prob = tf.sigmoid(logits).numpy()       # (B,3) вероятности 0..1
        y_true = y_true.numpy().astype(np.int32)  # (B,3) 0/1

        y_pred = (y_prob >= thresholds).astype(np.int32)

        B = y_true.shape[0]
        total_samples += B

        # exact-match: все 3 метки верны
        exact_match_correct += np.sum(np.all(y_pred == y_true, axis=1))

        # micro accuracy: по всем меткам вместе
        micro_correct += np.sum(y_pred == y_true)
        micro_total   += y_true.size  # B*3

        # per-head confusion
        for name, idx in [("blur",0), ("under",1), ("over",2)]:
            yt = y_true[:, idx]
            yp = y_pred[:, idx]

            tp = int(np.sum((yp == 1) & (yt == 1)))
            fp = int(np.sum((yp == 1) & (yt == 0)))
            fn = int(np.sum((yp == 0) & (yt == 1)))
            tn = int(np.sum((yp == 0) & (yt == 0)))

            counts[name]["tp"] += tp
            counts[name]["fp"] += fp
            counts[name]["fn"] += fn
            counts[name]["tn"] += tn

    # --- Вывод ---
    print("\n[VAL METRICS]")
    print("val samples:", total_samples)
    print(f"thresholds: blur={TH_BLUR} under={TH_UNDER} over={TH_OVER}")

    micro_acc = safe_div(micro_correct, micro_total)
    exact_acc = safe_div(exact_match_correct, total_samples)

    print(f"\nMicro-accuracy (по всем меткам вместе): {micro_acc:.4f}")
    print(f"Exact-match accuracy (все 3 метки сразу): {exact_acc:.4f}")

    for name in ["blur", "under", "over"]:
        tp = counts[name]["tp"]
        fp = counts[name]["fp"]
        fn = counts[name]["fn"]
        tn = counts[name]["tn"]
        prec, rec, f1, acc = compute_metrics_from_counts(tp, fp, fn, tn)
        support_pos = tp + fn
        support_neg = tn + fp

        print(f"\n[{name.upper()}]")
        print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"pos={support_pos} neg={support_neg}")
        print(f"precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
