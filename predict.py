import os
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
MODEL_PATH = "models/defect_classifier.h5"
TEST_IMAGE = "sample_test.jpeg"

os.makedirs("outputs", exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

img = cv2.imread(TEST_IMAGE)
if img is None:
    print("❌ Test image not found. Put an image named sample_test.jpg in the project root.")
    exit()

resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
normalized = resized.astype("float32") / 255.0
input_batch = np.expand_dims(normalized, axis=0)

pred = model.predict(input_batch)[0][0]
print("Raw prediction:", pred)

if pred >= 0.5:
    label = "OK"
    confidence = pred
else:
    label = "DEFECT"
    confidence = 1 - pred

print(f"Prediction: {label} ({confidence:.2f})")


# -----------------------------
# Annotate and Save Result
# -----------------------------
output_img = img.copy()

cv2.putText(
    output_img,
    f"{label} ({confidence:.2f})",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

output_path = os.path.join("outputs", "prediction_result.jpg")
cv2.imwrite(output_path, output_img)

print(f"✅ Saved result to: {output_path}")
