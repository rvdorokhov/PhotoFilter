import tensorflow as tf
import pandas as pd

IMG_H, IMG_W = 384, 512
BATCH = 4
VAL_CSV = "val.csv"                 # или "val_test.csv"
MODEL_PATH = "iq_multihd_savedmodel_1,5.keras"

AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)  # 0..255 (EfficientNet сам нормализует, раз у него есть rescaling)
    img = tf.image.resize(img, (IMG_H, IMG_W), method="bilinear")
    return img


def make_ds(csv_path):
    df = pd.read_csv(csv_path)
    paths = df["path"].astype(str).tolist()
    labels = df[["blur", "under", "over"]].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, y):
        img = decode_and_resize(path)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds


def main():
    # 1) Загружаем модель
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 2) Делаем val_ds
    val_ds = make_ds(VAL_CSV)

    # 3) Метрики accuracy по порогу 0.5
    acc_all = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    acc_blur = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    acc_under = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    acc_over = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

    # 4) Прогоняем val
    for x, y in val_ds:
        logits = model(x, training=False)
        prob = tf.sigmoid(logits)

        acc_all.update_state(y, prob)
        acc_blur.update_state(y[:, 0], prob[:, 0])
        acc_under.update_state(y[:, 1], prob[:, 1])
        acc_over.update_state(y[:, 2], prob[:, 2])

    print("VAL accuracy (all labels):", float(acc_all.result()))
    print("VAL accuracy blur:", float(acc_blur.result()))
    print("VAL accuracy under:", float(acc_under.result()))
    print("VAL accuracy over:", float(acc_over.result()))


if __name__ == "__main__":
    main()
