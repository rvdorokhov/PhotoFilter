# eval_two_val_sets.py
import numpy as np
import pandas as pd
import tensorflow as tf

# -----------------
# Настройки
# -----------------
IMG_H, IMG_W = 384, 512

MODEL_PATH = "iq_multihd_savedmodel_4heads.keras"

VAL_CSV       = "val.csv"        # обычный val для blur/under/over
NIGHT_VAL_CSV = "night_val.csv"  # отдельный val ТОЛЬКО для night

TH_BLUR  = 0.50
TH_UNDER = 0.50
TH_OVER  = 0.50
TH_NIGHT = 0.50

BATCH = 4
AUTOTUNE = tf.data.AUTOTUNE

# сколько примеров FP вывести в конце
FP_UNDER_SHOW = 50


# -----------------
# Data
# -----------------
def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)  # [0..255]
    img = tf.image.resize(img, (IMG_H, IMG_W), method="bilinear")
    return img

def make_ds_from_df(df, label_cols, batch=BATCH):
    paths = df["path"].astype(str).str.replace("\\", "/", regex=False).tolist()
    labels = df[label_cols].astype("float32").values

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

def update_counts(counts, yt, yp):
    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    counts["tp"] += tp
    counts["fp"] += fp
    counts["fn"] += fn
    counts["tn"] += tn

def print_head_metrics(name, c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec, rec, f1, acc = compute_metrics_from_counts(tp, fp, fn, tn)
    print(f"\n[{name.upper()}]")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} accuracy={acc:.4f}")


# -----------------
# main
# -----------------
def main():
    # GPU memory growth (чтобы не вылетало)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # --- Загружаем модель ---
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("\nLoaded model:", MODEL_PATH)

    # Проверим, что у модели 4 выхода
    dummy = tf.zeros((1, IMG_H, IMG_W, 3), tf.float32)
    out_dim = int(model(dummy, training=False).shape[-1])
    print("[INFO] Model output dim:", out_dim)
    if out_dim < 4:
        raise RuntimeError("Модель должна выдавать 4 логита (blur, under, over, night).")

    # ==========================================================
    # (A) Оценка blur/under/over на обычном val.csv
    # + сбор FP по under и статистики по night на этих FP
    # ==========================================================
    val_df = pd.read_csv(VAL_CSV)

    counts_3 = {
        "blur":  {"tp":0, "fp":0, "fn":0, "tn":0},
        "under": {"tp":0, "fp":0, "fn":0, "tn":0},
        "over":  {"tp":0, "fp":0, "fn":0, "tn":0},
    }

    # Собираем примеры FP по under:
    # список словарей: path, under_prob, night_prob
    under_fp_examples = []

    # Статистика: сколько FP по under всего, и сколько из них night_pred==1
    total_under_fp = 0
    total_under_fp_night_pred = 0

    total_samples = 0
    exact_match_correct = 0
    micro_correct = 0
    micro_total = 0

    thresholds_3 = np.array([TH_BLUR, TH_UNDER, TH_OVER], dtype=np.float32)

    paths_all = val_df["path"].astype(str).str.replace("\\", "/", regex=False).tolist()
    labels_all = val_df[["blur", "under", "over"]].astype("int32").values

    for start in range(0, len(paths_all), BATCH):
        end = min(start + BATCH, len(paths_all))
        batch_paths = paths_all[start:end]
        batch_y_true = labels_all[start:end]  # (B,3)

        batch_x = tf.stack([decode_and_resize(p) for p in batch_paths], axis=0)  # (B,H,W,3)

        logits4 = model(batch_x, training=False).numpy()  # (B,4)
        prob4 = tf.sigmoid(logits4).numpy()               # (B,4)

        prob3 = prob4[:, :3]  # blur/under/over
        y_pred = (prob3 >= thresholds_3).astype(np.int32)

        B = batch_y_true.shape[0]
        total_samples += B

        exact_match_correct += int(np.sum(np.all(y_pred == batch_y_true, axis=1)))
        micro_correct += int(np.sum(y_pred == batch_y_true))
        micro_total   += int(batch_y_true.size)

        update_counts(counts_3["blur"],  batch_y_true[:,0], y_pred[:,0])
        update_counts(counts_3["under"], batch_y_true[:,1], y_pred[:,1])
        update_counts(counts_3["over"],  batch_y_true[:,2], y_pred[:,2])

        # FP по under: pred=1, true=0
        fp_mask = (y_pred[:,1] == 1) & (batch_y_true[:,1] == 0)
        if np.any(fp_mask):
            idxs = np.where(fp_mask)[0]
            total_under_fp += int(len(idxs))

            # night_pred на этих FP (для статистики)
            night_prob_fp = prob4[idxs, 3]
            night_pred_fp = (night_prob_fp >= TH_NIGHT).astype(np.int32)
            total_under_fp_night_pred += int(np.sum(night_pred_fp == 1))

            # сохраняем несколько примеров для печати
            for j in idxs:
                if len(under_fp_examples) < FP_UNDER_SHOW:
                    under_fp_examples.append({
                        "path": batch_paths[j],
                        "under_prob": float(prob4[j, 1]),
                        "night_prob": float(prob4[j, 3]),
                        "night_pred": int(prob4[j, 3] >= TH_NIGHT),
                    })

    print("\n==============================")
    print("[VAL.CSV] blur/under/over")
    print("==============================")
    print("val samples:", total_samples)
    print(f"thresholds: blur={TH_BLUR} under={TH_UNDER} over={TH_OVER} night={TH_NIGHT}")

    micro_acc = safe_div(micro_correct, micro_total)
    exact_acc = safe_div(exact_match_correct, total_samples)
    print(f"\nMicro-accuracy (3 метки): {micro_acc:.4f}")
    print(f"Exact-match accuracy (3 метки): {exact_acc:.4f}")

    print_head_metrics("blur",  counts_3["blur"])
    print_head_metrics("under", counts_3["under"])
    print_head_metrics("over",  counts_3["over"])

    # ---- статистика по night на FP under ----
    print("\n===================================")
    print("[UNDER FP] night-статистика на FP-сценах")
    print("===================================")
    print("Всего FP по under:", total_under_fp)
    if total_under_fp > 0:
        pct = 100.0 * total_under_fp_night_pred / total_under_fp
        print(f"Night_pred==1 среди FP under: {total_under_fp_night_pred}/{total_under_fp} = {pct:.2f}%")
        print("Интерпретация: если в приложении отключать under при night_pred==1,")
        print("то теоретически можно снять примерно этот процент ложных срабатываний under (на этом val).")
    else:
        print("FP по under не найдено (при заданном пороге).")

    # ==========================================================
    # (B) Оценка ТОЛЬКО night на night_val.csv
    # ==========================================================
    night_df = pd.read_csv(NIGHT_VAL_CSV)
    if "night" not in night_df.columns:
        raise ValueError("night_val.csv должен содержать колонку 'night'.")

    night_ds = make_ds_from_df(night_df, ["night"], batch=BATCH)
    counts_night = {"tp":0, "fp":0, "fn":0, "tn":0}
    total_night_samples = 0

    for x, y_true_n in night_ds:
        logits4 = model(x, training=False)                # (B,4)
        night_prob = tf.sigmoid(logits4[:, 3]).numpy()    # (B,)
        night_pred = (night_prob >= TH_NIGHT).astype(np.int32)

        y_true_n = y_true_n.numpy().astype(np.int32).reshape(-1)  # (B,)
        update_counts(counts_night, y_true_n, night_pred)
        total_night_samples += int(y_true_n.shape[0])

    print("\n===================================")
    print("[NIGHT_VAL.CSV] ONLY night head")
    print("===================================")
    print("night_val samples:", total_night_samples)
    print(f"threshold: night={TH_NIGHT}")
    print_head_metrics("night", counts_night)

    # ==========================================================
    # (C) В конце: список (до 50) FP under + night_prob
    # ==========================================================
    print("\n===================================")
    print(f"[UNDER FP] примеры (до {FP_UNDER_SHOW})")
    print("===================================")
    if under_fp_examples:
        for ex in under_fp_examples[:FP_UNDER_SHOW]:
            # можно печатать компактно
            print(f"{ex['path']}  | under_prob={ex['under_prob']:.3f}  night_prob={ex['night_prob']:.3f}  night_pred={ex['night_pred']}")
    else:
        print("FP по under не найдено (при заданном пороге).")


if __name__ == "__main__":
    main()
