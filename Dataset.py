#Data generator
import math
import numpy as np
import cv2

def toDistrs(Xs):
    returnval = [[0, 0, 0, 0, 0] for i in range(len(Xs[0]))]

    for X in Xs:
        for i in range(len(X)):
            returnval[i][X[i]] += 1.0 / len(Xs)

    return returnval

def dataset_generator(batch_size, sample_space_size, uniform=False, exponent=1,seed=1234):
    # generate the distribution over X (5 elements with a space of 5 values each for the sizes)
    pr_x = []
    Xs = None

    np.random.seed(seed)


    for idx in range(5):
        if uniform:
            raw = [1 for i in range(5)]
        else:
            raw = [math.exp(exponent * np.random.random_sample()) for i in range(5)]

        pr_x.append([float(i) / sum(raw) for i in raw])
        tmp = np.random.choice(5, (sample_space_size, 1), p=pr_x[idx])

        if idx == 0:
            Xs = tmp
        else:
            Xs = np.concatenate((Xs, tmp), axis=1)

    while True:
        images = np.zeros((batch_size, 512, 512, 3), dtype=np.float32)
        labels = np.zeros((batch_size, 5, 5), dtype=np.float32)

        for i in range(batch_size):
            xs = np.random.binomial(100, 0.25, (1, 5))[0]
            xs *= 10
            ys = np.random.binomial(100, 0.25, (1, 5))[0]
            ys *= 10

            radius_sizes = [50, 60, 70, 80, 90]
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (191, 64, 191), (0, 255, 255)]
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            img.fill(255)

            sizes_label_mapping = {50: 0, 60: 1, 70: 2, 80: 3, 90: 4}

            for j in range(5):
                x, y = xs[j], ys[j]
                # Get the corresponding radius size from the distribution
                radius = radius_sizes[Xs[i][j]]
                color = colors[j]
                cv2.circle(img, (x, y), radius, color, -1)
                label = np.zeros(5, dtype=np.int8)
                label[sizes_label_mapping[radius]] = 1
                labels[i, j] = label

            img = img / 255.0
            images[i] = img

        yield images, labels
