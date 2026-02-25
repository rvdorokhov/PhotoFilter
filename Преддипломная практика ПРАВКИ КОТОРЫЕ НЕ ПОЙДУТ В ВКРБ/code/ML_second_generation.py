import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt

IMG_H, IMG_W = 384, 512  # tf: (H,W)
BATCH = 16
EPOCHS_HEADS = 5
EPOCHS_FT = 5

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"

AUTOTUNE = tf.data.AUTOTUNE


class PlotCurvesCallback(tf.keras.callbacks.Callback):
    """
    Рисует:
      - общий loss (train/val)
      - loss по головам (blur/under/over, train/val)  <-- ДОБАВИЛИ
      - AUC общий + по головам (train/val)

    Память почти не расходует: храним только списки длиной = числу эпох.
    """
    def __init__(self, out_path="training_curves.png"):
        super().__init__()
        self.out_path = out_path

        # Ось X: глобальные эпохи (не сбрасываются между двумя fit())
        self.global_epoch = 0
        self.epochs = []

        # ---- Общий loss ----
        self.loss = []
        self.val_loss = []

        # ---- Loss по головам (BCE как метрика) ----
        self.bce_blur = []
        self.val_bce_blur = []
        self.bce_under = []
        self.val_bce_under = []
        self.bce_over = []
        self.val_bce_over = []

        # ---- AUC общий ----
        self.auc = []
        self.val_auc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Глобальная эпоха: 1..N по всем fit()
        self.global_epoch += 1
        self.epochs.append(self.global_epoch)

        # ---- Общий loss ----
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))

        # ---- Loss по головам ----
        self.bce_blur.append(logs.get("bce_blur"))
        self.val_bce_blur.append(logs.get("val_bce_blur"))

        self.bce_under.append(logs.get("bce_under"))
        self.val_bce_under.append(logs.get("val_bce_under"))

        self.bce_over.append(logs.get("bce_over"))
        self.val_bce_over.append(logs.get("val_bce_over"))

        # ---- AUC общий ----
        self.auc.append(logs.get("auc"))
        self.val_auc.append(logs.get("val_auc"))

        # ---- Рисуем и перезаписываем файл ----
        fig = plt.figure(figsize=(7.5, 3), dpi=120)

        # ===== Loss (общий + по головам) =====
        ax1 = fig.add_subplot(1, 2, 1)

        # общий loss
        ax1.plot(self.epochs, self.loss, label="train loss")
        ax1.plot(self.epochs, self.val_loss, label="val loss")

        # head losses (если они есть в logs; иначе будут None -> matplotlib может ругаться)
        # поэтому рисуем только если хотя бы одно значение не None
        if any(v is not None for v in self.bce_blur):
            ax1.plot(self.epochs, self.bce_blur, label="train bce_blur", linestyle="--")
            ax1.plot(self.epochs, self.val_bce_blur, label="val bce_blur", linestyle="--")

        if any(v is not None for v in self.bce_under):
            ax1.plot(self.epochs, self.bce_under, label="train bce_under", linestyle=":")
            ax1.plot(self.epochs, self.val_bce_under, label="val bce_under", linestyle=":")

        if any(v is not None for v in self.bce_over):
            ax1.plot(self.epochs, self.bce_over, label="train bce_over", linestyle="-.")
            ax1.plot(self.epochs, self.val_bce_over, label="val bce_over", linestyle="-.")

        ax1.set_title("Loss")
        ax1.set_xlabel("epoch")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=6)

        # ===== AUC (общий + по головам) =====
        ax2 = fig.add_subplot(1, 2, 2)

        ax2.plot(self.epochs, self.auc, label="train all")
        ax2.plot(self.epochs, self.val_auc, label="val all")

        ax2.set_title("AUC")
        ax2.set_xlabel("epoch")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower right", fontsize=6)

        fig.tight_layout()
        fig.savefig(self.out_path)
        plt.close(fig)  # важно: закрываем фигуру, чтобы не копить память




def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)  # jpg/png/bmp ок
    img = tf.cast(img, tf.float32)    # [0..255]
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
    x = tf.keras.layers.Dropout(0.2, name="neck_dropout")(x)

    def head_exposure(name: str, x):
        # under/over: 32 -> 16 -> 1
        h = tf.keras.layers.Dense(32, activation="relu", name=f"{name}_fc1")(x)
        h = tf.keras.layers.Dropout(0.1, name=f"{name}_drop1")(h)
        h = tf.keras.layers.Dense(16, activation="relu", name=f"{name}_fc2")(h)
        out = tf.keras.layers.Dense(1, name=f"logit_{name}")(h)
        return out

    def head_blur(name: str, x):
        # blur: 64 -> 32 -> 16 -> 1
        h = tf.keras.layers.Dense(64, activation="relu", name=f"{name}_fc1")(x)
        h = tf.keras.layers.Dropout(0.15, name=f"{name}_drop1")(h)
        h = tf.keras.layers.Dense(32, activation="relu", name=f"{name}_fc2")(h)
        h = tf.keras.layers.Dense(16, activation="relu", name=f"{name}_fc3")(h)
        out = tf.keras.layers.Dense(1, name=f"logit_{name}")(h)
        return out

    logit_blur  = head_blur("blur", x)
    logit_under = head_exposure("under", x)
    logit_over  = head_exposure("over", x)

    logits = tf.keras.layers.Concatenate(axis=1, name="logits")(
        [logit_blur, logit_under, logit_over]
    )

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

    def reset_state(self):
        self.auc.reset_state()

class PerLabelBCE(tf.keras.metrics.Metric):
    """BCE (from_logits=True) по одному столбцу: 0=blur, 1=under, 2=over"""
    def __init__(self, label_index: int, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_index = label_index
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true[:, self.label_index], tf.float32)
        lg = y_pred[:, self.label_index]        # (B,)

        # BCE для логитов: -[ y*log(sigmoid(l)) + (1-y)*log(1-sigmoid(l)) ]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=lg)  # (B,)

        self.mean.update_state(loss, sample_weight=sample_weight)

    def result(self):
        return self.mean.result()

    def reset_state(self):
        self.mean.reset_state()

def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            MultiLabelAUC(name="auc"),

            # <<< loss по головам (как метрики)
            PerLabelBCE(0, name="bce_blur"),
            PerLabelBCE(1, name="bce_under"),
            PerLabelBCE(2, name="bce_over"),
        ],
    )


def main():
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    print(tf.config.list_physical_devices())
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
    
    train_ds = make_ds(TRAIN_CSV, shuffle=True)
    val_ds = make_ds(VAL_CSV, shuffle=False)

    # 1) формы
    x_batch, y_batch = next(iter(train_ds.take(1)))
    print("\n[DEBUG] Train batch shapes:")
    print("  x:", x_batch.shape, x_batch.dtype)
    print("  y:", y_batch.shape, y_batch.dtype)

    # 2) диапазон значений пикселей (после decode + convert_image_dtype)
    x_min = tf.reduce_min(x_batch).numpy()
    x_max = tf.reduce_max(x_batch).numpy()
    print("[DEBUG] Pixel range after preprocessing: min =", x_min, "max =", x_max)

    model, backbone = build_model()

    print("\n[DEBUG] Model summary:")
    model.summary()

    plot_cb = PlotCurvesCallback("training_curves.png")
    csv_cb = tf.keras.callbacks.CSVLogger("train_log.csv", append=True)
    callbacks = [plot_cb, csv_cb]

    # Этап 1: учим только головы
    backbone.trainable = False
    compile_model(model, lr=1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEADS, callbacks=callbacks)

    # Этап 2: fine-tune (можно разморозить всё или только верхние слои)
    # Часто стабильнее размораживать не всё сразу, а только верх:
    backbone.trainable = True

    # заморозить BatchNorm
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # заморозить нижнюю часть
    for layer in backbone.layers[:-40]:
        layer.trainable = False


    compile_model(model, lr=5e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=callbacks)

    model.save("iq_multihd_savedmodel.keras")
    print("Saved model to iq_multihd_savedmodel")

if __name__ == "__main__":
    main()
