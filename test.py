import tensorflow as tf
m = tf.keras.models.load_model("iq_multihd_savedmodel_2.keras", compile=False)
m.summary()
