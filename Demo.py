import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Dataset import dataset_generator

model = tf.keras.models.load_model('my_model_v2_trained.h5', compile=False)
# Get a sample batch from the test generator

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics="categorical_accuracy")

test_generator = dataset_generator(25,25)
test_images, test_labels = next(test_generator)

# Predict the probabilities
predicted_probabilities = model.predict(test_images)

num_samples = 10
input_shape = (512, 512, 3)
test_generator = dataset_generator(25, 25)
X_test, y_test = next(test_generator)

# Evaluate the model on the test data
loss, *metrics = model.evaluate(X_test, y_test)


print(predicted_probabilities[0])

# Convert probabilities to class labels
predicted_labels = np.argmax(predicted_probabilities, axis=-1)

# Show the images, true labels, and predicted labels
n_images_to_show = 5

for i in range(n_images_to_show):
    plt.imshow(test_images[i])
    plt.title(f"True labels: {np.argmax(test_labels[i], axis=-1)}\nPredicted labels: {predicted_labels[i]}")
    plt.axis("off")
    plt.show()

