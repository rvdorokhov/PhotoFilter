import tensorflow as tf


# ---------------------------
# Твои пользовательские классы
# ---------------------------
class MultiLabelAUC(tf.keras.metrics.Metric):
    def __init__(self, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC(multi_label=True, num_labels=4)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_prob = tf.sigmoid(y_pred)
        self.auc.update_state(y_true, y_prob)

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()


class BCEPerLabel(tf.keras.losses.Loss):
    def __init__(self, name="bce_per_label"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)


class PerLabelBCE(tf.keras.metrics.Metric):
    def __init__(self, label_index: int, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_index = label_index
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true[:, self.label_index], tf.float32)
        lg = y_pred[:, self.label_index]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=lg)

        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            if len(sw.shape) == 2:
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


# ---------------------------
# Экспорт
# ---------------------------
MODEL_PATH = "iq_multihd_savedmodel_4heads.keras"
EXPORT_DIR = "iq_multihd_savedmodel"


def main():
    print(f"Loading model from: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "MultiLabelAUC": MultiLabelAUC,
            "BCEPerLabel": BCEPerLabel,
            "PerLabelBCE": PerLabelBCE,
        },
        compile=False,  # для экспорта компиляция не нужна
    )

    print("Model loaded successfully.")
    model.summary()

    print(f"Exporting to SavedModel: {EXPORT_DIR}")
    model.export(EXPORT_DIR)

    print("Done.")
    print(f"SavedModel exported to: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
