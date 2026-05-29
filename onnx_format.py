import tensorflow as tf


# ---------------------------
# Пользовательские классы
# ---------------------------
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
    """BCE по одному выходу: 0=blur, 1=under, 2=over"""
    def __init__(self, label_index: int, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_index = label_index
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true[:, self.label_index], tf.float32)
        lg = y_pred[:, self.label_index]

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=lg)

        self.mean.update_state(loss, sample_weight=sample_weight)

    def result(self):
        return self.mean.result()

    def reset_state(self):
        self.mean.reset_state()


# ---------------------------
# Экспорт
# ---------------------------
MODEL_PATH = "iq_multihd_savedmodel_2.keras"
EXPORT_DIR = "iq_multihd_savedmodel_3heads"


def main():
    print(f"Loading model from: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "MultiLabelAUC": MultiLabelAUC,
            "PerLabelBCE": PerLabelBCE,
        },
        compile=False
    )

    print("Model loaded successfully.")
    model.summary()

    print(f"Exporting to SavedModel: {EXPORT_DIR}")
    model.export(EXPORT_DIR)

    print("Done.")
    print(f"SavedModel exported to: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
