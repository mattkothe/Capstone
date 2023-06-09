import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Dataset import dataset_generator

model = tf.keras.models.load_model('capstone_model.hdf5', compile=False)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics="accuracy")

test_generator = dataset_generator(25,25)
test_images, test_labels = next(test_generator)

predicted_probabilities = model.predict(test_images)

num_samples = 10
input_shape = (512, 512, 3)
test_generator = dataset_generator(25, 25)
X_test, y_test = next(test_generator)

loss, *metrics = model.evaluate(X_test, y_test)


print(predicted_probabilities[0])

predicted_labels = np.argmax(predicted_probabilities, axis=-1)

n_images_to_show = 5

for i in range(n_images_to_show):
    plt.imshow(test_images[i])
    plt.title(f"True labels: {np.argmax(test_labels[i], axis=-1)}\nPredicted labels: {predicted_labels[i]}")
    plt.axis("off")
    plt.show()

