import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

IMG_H, IMG_W = 384, 512  # tf: (H,W)
BATCH = 16
EPOCHS_HEADS = 5
EPOCHS_FT = 5

TRAIN_CSV = "train_test.csv"
VAL_CSV = "val_test.csv"

# night-датасет (появится позже)
EPOCHS_NIGHT = 5
NIGHT_TRAIN_CSV = "night_train_test.csv"   # CSV: path, night (0/1)
NIGHT_VAL_CSV   = "night_val_test.csv"

AUTOTUNE = tf.data.AUTOTUNE


class PlotCurvesCallback(tf.keras.callbacks.Callback):
    """
    Рисует:
      - общий loss (train/val)
      - loss по головам (blur/under/over/night, train/val)
      - AUC общий (по всем меткам)

    Память почти не расходует: храним только списки длиной = числу эпох.
    """
    def __init__(self, out_path="training_curves.png"):
        super().__init__()
        self.out_path = out_path

        self.global_epoch = 0
        self.epochs = []

        self.loss = []
        self.val_loss = []

        self.bce_blur = []
        self.val_bce_blur = []
        self.bce_under = []
        self.val_bce_under = []
        self.bce_over = []
        self.val_bce_over = []
        self.bce_night = []
        self.val_bce_night = []

        self.auc = []
        self.val_auc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        self.global_epoch += 1
        self.epochs.append(self.global_epoch)

        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))

        self.bce_blur.append(logs.get("bce_blur"))
        self.val_bce_blur.append(logs.get("val_bce_blur"))

        self.bce_under.append(logs.get("bce_under"))
        self.val_bce_under.append(logs.get("val_bce_under"))

        self.bce_over.append(logs.get("bce_over"))
        self.val_bce_over.append(logs.get("val_bce_over"))

        self.bce_night.append(logs.get("bce_night"))
        self.val_bce_night.append(logs.get("val_bce_night"))

        self.auc.append(logs.get("auc"))
        self.val_auc.append(logs.get("val_auc"))

        fig = plt.figure(figsize=(14, 4), dpi=120)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(self.epochs, self.loss, label="train loss")
        ax1.plot(self.epochs, self.val_loss, label="val loss")

        if any(v is not None for v in self.bce_blur):
            ax1.plot(self.epochs, self.bce_blur, label="train bce_blur", linestyle="--")
            ax1.plot(self.epochs, self.val_bce_blur, label="val bce_blur", linestyle="--")

        if any(v is not None for v in self.bce_under):
            ax1.plot(self.epochs, self.bce_under, label="train bce_under", linestyle=":")
            ax1.plot(self.epochs, self.val_bce_under, label="val bce_under", linestyle=":")

        if any(v is not None for v in self.bce_over):
            ax1.plot(self.epochs, self.bce_over, label="train bce_over", linestyle="-.")
            ax1.plot(self.epochs, self.val_bce_over, label="val bce_over", linestyle="-.")

        if any(v is not None for v in self.bce_night):
            ax1.plot(self.epochs, self.bce_night, label="train bce_night", linestyle=(0, (3, 1, 1, 1)))
            ax1.plot(self.epochs, self.val_bce_night, label="val bce_night", linestyle=(0, (3, 1, 1, 1)))

        ax1.set_title("Loss")
        ax1.set_xlabel("epoch")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=6)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(self.epochs, self.auc, label="train all")
        ax2.plot(self.epochs, self.val_auc, label="val all")
        ax2.set_title("AUC")
        ax2.set_xlabel("epoch")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=6)

        fig.tight_layout()
        fig.tight_layout(rect=[0, 0, 0.78, 1])  # оставляем место справа под легенду
        fig.savefig(self.out_path)
        plt.close(fig)


def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)  # jpg/png/bmp ок
    img = tf.cast(img, tf.float32)    # [0..255]
    img = tf.image.resize(img, (IMG_H, IMG_W), method="bilinear")
    return img


def make_ds(csv_path, shuffle=False):
    """
    Датасет качества:
      y = [blur, under, over, night(=0)]
      sample_weight = [1, 1, 1, 0]  -> night не участвует в loss на этапах 1-2
    """
    df = pd.read_csv(csv_path)
    paths = df["path"].astype(str).tolist()

    y3 = df[["blur", "under", "over"]].astype("float32").values  # (N,3)
    night0 = tf.zeros((len(paths), 1), dtype=tf.float32).numpy() # (N,1)
    y4 = tf.concat([y3, night0], axis=1).numpy().astype("float32")  # (N,4)

    # веса по компонентам: blur/under/over считаем, night игнорируем
    sw_row = tf.constant([1., 1., 1., 0.], dtype=tf.float32).numpy()    # (4,)
    sw = np.tile(sw_row, (len(paths), 1)).astype("float32")             # (N,4)

    ds = tf.data.Dataset.from_tensor_slices((paths, y4, sw))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 20000), reshuffle_each_iteration=True)

    def _map_fn(path, y, sw):
        img = decode_and_resize(path)
        return img, y, sw  # <- важно: три значения

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds


def make_ds_night(csv_path, shuffle=False):
    """
    Датасет night:
      y = [0,0,0, night]
      sample_weight = [0,0,0,1] -> считаем только night в loss на этапе 3
    """
    df = pd.read_csv(csv_path)
    paths = df["path"].astype(str).tolist()

    night = df["night"].astype("float32").values.reshape(-1, 1)          # (N,1)
    zeros3 = tf.zeros((len(paths), 3), dtype=tf.float32).numpy()         # (N,3)
    y4 = tf.concat([zeros3, night], axis=1).numpy().astype("float32")    # (N,4)

    # веса по компонентам: только night
    sw_row = tf.constant([0., 0., 0., 1.], dtype=tf.float32).numpy()
    sw = np.tile(sw_row, (len(paths), 1)).astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, y4, sw))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 20000), reshuffle_each_iteration=True)

    def _map_fn(path, y, sw):
        img = decode_and_resize(path)
        return img, y, sw

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
    x = tf.keras.layers.Dropout(0.2, name="neck_dropout")(x)

    def head_exposure(name: str, x_):
        h = tf.keras.layers.Dense(32, activation="relu", name=f"{name}_fc1")(x_)
        h = tf.keras.layers.Dropout(0.1, name=f"{name}_drop1")(h)
        h = tf.keras.layers.Dense(16, activation="relu", name=f"{name}_fc2")(h)
        out = tf.keras.layers.Dense(1, name=f"logit_{name}")(h)
        return out

    def head_blur(name: str, x_):
        h = tf.keras.layers.Dense(64, activation="relu", name=f"{name}_fc1")(x_)
        h = tf.keras.layers.Dropout(0.15, name=f"{name}_drop1")(h)
        h = tf.keras.layers.Dense(32, activation="relu", name=f"{name}_fc2")(h)
        h = tf.keras.layers.Dense(16, activation="relu", name=f"{name}_fc3")(h)
        out = tf.keras.layers.Dense(1, name=f"logit_{name}")(h)
        return out

    def head_night(x_):
        # night: 64 -> 32 -> 16 -> 1 (как blur)
        h = tf.keras.layers.Dense(64, activation="relu", name="night_fc1")(x_)
        h = tf.keras.layers.Dropout(0.15, name="night_drop1")(h)
        h = tf.keras.layers.Dense(32, activation="relu", name="night_fc2")(h)
        h = tf.keras.layers.Dense(16, activation="relu", name="night_fc3")(h)
        out = tf.keras.layers.Dense(1, name="logit_night")(h)
        return out

    logit_blur  = head_blur("blur", x)
    logit_under = head_exposure("under", x)
    logit_over  = head_exposure("over", x)
    logit_night = head_night(x)

    logits = tf.keras.layers.Concatenate(axis=1, name="logits")(
        [logit_blur, logit_under, logit_over, logit_night]
    )

    model = tf.keras.Model(inputs=inp, outputs=logits, name="iq_multihd")
    return model, backbone


class MultiLabelAUC(tf.keras.metrics.Metric):
    def __init__(self, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC(multi_label=True, num_labels=4)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_prob = tf.sigmoid(y_pred)
        # sample_weight игнорируем, чтобы не было конфликтов формы (B,4)
        self.auc.update_state(y_true, y_prob)

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()


class BCEPerLabel(tf.keras.losses.Loss):
    """Поэлементная BCE по логитам: возвращает (B,4), чтобы работал sample_weight (B,4)."""
    def __init__(self, name="bce_per_label"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)  # (B,4)


class PerLabelBCE(tf.keras.metrics.Metric):
    """BCE (from_logits=True) по одному столбцу, с учётом sample_weight[:, label_index]"""
    def __init__(self, label_index: int, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_index = label_index
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true[:, self.label_index], tf.float32)
        lg = y_pred[:, self.label_index]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=lg)  # (B,)

        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            # если sw формы (B,4) -> берём нужный столбец
            if tf.rank(sw) == 2:
                sw = sw[:, self.label_index]
            self.total.assign_add(tf.reduce_sum(loss * sw))
            self.count.assign_add(tf.reduce_sum(sw))
        else:
            self.total.assign_add(tf.reduce_sum(loss))
            self.count.assign_add(tf.cast(tf.size(loss), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=BCEPerLabel(),
        metrics=[
            MultiLabelAUC(name="auc"),
            PerLabelBCE(0, name="bce_blur"),
            PerLabelBCE(1, name="bce_under"),
            PerLabelBCE(2, name="bce_over"),
            PerLabelBCE(3, name="bce_night"),
        ],
    )


# ---------------------------
# Заморозка/разморозка night
# ---------------------------
def freeze_night_head(model):
    """Заморозить слои night, чтобы они НЕ обучались на этапах 1-2."""
    for layer in model.layers:
        if layer.name.startswith("night_") or layer.name == "logit_night":
            layer.trainable = False


def unfreeze_night_head(model):
    """Разморозить слои night, чтобы они обучались на этапе 3."""
    for layer in model.layers:
        if layer.name.startswith("night_") or layer.name == "logit_night":
            layer.trainable = True


def freeze_everything_except_night(model):
    """На этапе 3 фиксируем всё и обучаем только night."""
    for layer in model.layers:
        layer.trainable = False
    unfreeze_night_head(model)


def main():
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    print(tf.config.list_physical_devices())
    print("Num GPUs Available: ", len(physical_devices))
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_ds = make_ds(TRAIN_CSV, shuffle=True)
    val_ds = make_ds(VAL_CSV, shuffle=False)

    x_batch, y_batch, sw_batch = next(iter(train_ds.take(1)))
    print("\n[DEBUG] Train batch shapes:")
    print("  x:", x_batch.shape, x_batch.dtype)
    print("  y:", y_batch.shape, y_batch.dtype)
    print("  sw:", sw_batch.shape, sw_batch.dtype)

    x_min = tf.reduce_min(x_batch).numpy()
    x_max = tf.reduce_max(x_batch).numpy()
    print("[DEBUG] Pixel range after preprocessing: min =", x_min, "max =", x_max)

    model, backbone = build_model()

    print("\n[DEBUG] Model summary:")
    model.summary()

    plot_cb = PlotCurvesCallback("training_curves.png")
    csv_cb = tf.keras.callbacks.CSVLogger("train_log.csv", append=True)
    callbacks = [plot_cb, csv_cb]

    # ===== Этап 1: учим только головы качества (night заморожена) =====
    backbone.trainable = False
    freeze_night_head(model)  # <-- ВАЖНО: night не обучается на этапах 1-2

    compile_model(model, lr=1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEADS, callbacks=callbacks)

    # ===== Этап 2: fine-tune верхних слоёв основы (night всё ещё заморожена) =====
    backbone.trainable = True

    # заморозить BatchNorm
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # заморозить нижнюю часть
    for layer in backbone.layers[:-40]:
        layer.trainable = False

    freeze_night_head(model)  # <-- на всякий случай повторим (после изменения trainable)

    compile_model(model, lr=5e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=callbacks)

    # ===== Этап 3: обучаем только night, если датасет подготовлен =====
    if os.path.exists(NIGHT_TRAIN_CSV) and os.path.exists(NIGHT_VAL_CSV):
        print("\n[INFO] Найдены CSV для night -> старт обучения night-головы.")
        night_train_ds = make_ds_night(NIGHT_TRAIN_CSV, shuffle=True)
        night_val_ds   = make_ds_night(NIGHT_VAL_CSV, shuffle=False)

        freeze_everything_except_night(model)
        compile_model(model, lr=1e-3)
        model.fit(
            night_train_ds,
            validation_data=night_val_ds,
            epochs=EPOCHS_NIGHT,
            callbacks=callbacks
        )
    else:
        print(
            "\n[INFO] CSV для night не найдены -> этап night пропущен.\n"
            "Ожидаемый формат:\n"
            "  night_train.csv: path,night\n"
            "  night_val.csv:   path,night\n"
        )

    model.save("iq_multihd_savedmodel_4heads.keras")
    print("Saved model to iq_multihd_savedmodel_4heads.keras")


if __name__ == "__main__":
    main()
