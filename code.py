import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train_cat, epochs=5, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
# Predict on test images
predictions = model.predict(x_test)

# Show a sample prediction
index = 1234  # Pick any test index
plt.imshow(x_test[index], cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[index])}, Actual: {y_test[index]}")
plt.show()
!pip install gradio
import gradio as gr
import numpy as np # Import numpy if not already imported

def recognize_digit(img):
    # Ensure image is a NumPy array and has the correct shape
    img = np.array(img) # Convert to numpy array if it's not already
    img = img.reshape(1, 28, 28)
    img = img / 255.0
    # Assuming 'model' is defined elsewhere and accessible in this scope
    pred = model.predict(img).argmax()
    return f"Predicted Digit: {pred}"

gr.Interface(fn=recognize_digit,
             inputs=gr.Image(width=28, height=28, image_mode='L'), # Removed invert_colors
             outputs="text",
             title="🖊️ Handwritten Digit Recognizer",
             description="Draw a digit (0–9) and let the AI recognize it!"
).launch()
# Step 5: Gradio Interface
def recognize_digit(img):
    # Resize to 28x28 if needed
    img = tf.image.resize(img, [28, 28])
    img = tf.image.rgb_to_grayscale(img)
    # Invert colors manually if needed, since invert_colors is removed from gr.Image
    # img = 1.0 - img # Uncomment this line if you need to invert the colors
    img = img.numpy().reshape(1, 28, 28)
    img = img / 255.0
    pred = model.predict(img).argmax()
    return f"Predicted Digit: {pred}"

gr.Interface(
    fn=recognize_digit,
    # Removed 'invert_colors=True' as it's not a valid argument for gr.Image
    # Removed 'source="canvas"' as it is not a valid argument for gr.Image
    inputs=gr.Image(width=28, height=28, image_mode='L'),
    outputs="text",
    title="🖊️ Handwritten Digit Recognizer",
    description="Draw a digit (0–9) and let the AI recognize it!"
).launch()
