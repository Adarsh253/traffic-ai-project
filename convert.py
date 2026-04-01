import tensorflow as tf

# 1. Load your old file (make sure the path matches your folder exactly)
# If your model is in a folder called 'model', use "model/mobilenetv2_traffic.h5"
model = tf.keras.models.load_model("model/mobilenetv2_traffic.h5", compile=False)

# 2. Save it as the new version
model.save("model/mobilenetv2_traffic.keras")

print("Success! You should now see mobilenetv2_traffic.keras in your model folder.")