import tensorflow as tf

model = tf.keras.models.load_model("liveness.keras")
model.save("liveness_compatible.h5")
