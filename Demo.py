import numpy as np
import matplotlib.pyplot as plt

# Get a sample batch from the test generator
test_images, test_labels = next(test_generator)

# Predict the probabilities
predicted_probabilities = model.predict(test_images)

# Convert probabilities to class labels
predicted_labels = np.argmax(predicted_probabilities, axis=-1)

# Show the images, true labels, and predicted labels
n_images_to_show = 5

for i in range(n_images_to_show):
    plt.imshow(test_images[i])
    plt.title(f"True labels: {np.argmax(test_labels[i], axis=-1)}\nPredicted labels: {predicted_labels[i]}")
    plt.axis("off")
    plt.show()