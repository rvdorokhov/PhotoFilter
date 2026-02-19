import tensorflow as tf
import pandas as pd

IMG_H, IMG_W = 384, 512  # tf: (H,W)
BATCH = 16
EPOCHS_HEADS = 5
EPOCHS_FT = 5

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"

AUTOTUNE = tf.data.AUTOTUNE

def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)  # jpg/png/bmp ок
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0..1]
    img = tf.image.resize(img, (IMG_H, IMG_W), method="bilinear")
    return img

def make_ds(csv_path, shuffle=False):
    df = pd.read_csv(csv_path)
    paths = df["path"].astype(str).tolist()
    labels = df[["blur", "under", "over"]].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 20000), reshuffle_each_iteration=True)

    def _map_fn(path, y):
        img = decode_and_resize(path)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

def build_model():
    inp = tf.keras.Input(shape=(IMG_H, IMG_W, 3))

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inp
    )
    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # 3 головы -> логиты (без sigmoid)
    logit_blur  = tf.keras.layers.Dense(1, name="logit_blur")(x)
    logit_under = tf.keras.layers.Dense(1, name="logit_under")(x)
    logit_over  = tf.keras.layers.Dense(1, name="logit_over")(x)

    logits = tf.keras.layers.Concatenate(axis=1, name="logits")([logit_blur, logit_under, logit_over])

    model = tf.keras.Model(inputs=inp, outputs=logits, name="iq_multihd")
    return model, backbone

# Метрика: AUC по трём лейблам (работает с вероятностями, поэтому применим sigmoid внутри)
class MultiLabelAUC(tf.keras.metrics.Metric):
    def __init__(self, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC(multi_label=True, num_labels=3)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_prob = tf.sigmoid(y_pred)
        self.auc.update_state(y_true, y_prob, sample_weight=sample_weight)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()

def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[MultiLabelAUC()],
    )

def main():
    train_ds = make_ds(TRAIN_CSV, shuffle=True)
    val_ds = make_ds(VAL_CSV, shuffle=False)

    model, backbone = build_model()

    # Этап 1: учим только головы
    backbone.trainable = False
    compile_model(model, lr=1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEADS)

    # Этап 2: fine-tune (можно разморозить всё или только верхние слои)
    backbone.trainable = True
    # Часто стабильнее размораживать не всё сразу, а только верх:
    # for layer in backbone.layers[:-40]:
    #     layer.trainable = False

    compile_model(model, lr=1e-4)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT)

    model.save("iq_multihd_savedmodel")
    print("Saved model to iq_multihd_savedmodel")

if __name__ == "__main__":
    main()
