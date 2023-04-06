import tensorflow as tf
from keras_tuner import Hyperband, HyperModel
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Lambda
from keras.layers import Dropout
from keras.regularizers import l1, l2, l1_l2
from Dataset import dataset_generator

class CNNHyperModelAll(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        def normalize_probability_distributions(x):
            x_reshaped = tf.reshape(x, (-1, 5))
            probabilities = tf.nn.softmax(x_reshaped)
            return tf.reshape(probabilities, (-1, 5, 5))

        model = tf.keras.Sequential([
            Conv2D(filters=hp.Choice('conv1_filters', values=[32, 64], default=32),
                   kernel_size=hp.Choice('conv1_kernel_size', values=[1, 5], default=5),
                   activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=hp.Choice('pool1_size', values=[1, 5], default=5)),
            Conv2D(filters=hp.Choice('conv2_filters', values=[64, 128], default=64),
                   kernel_size=hp.Choice('conv2_kernel_size', values=[1, 5], default=5),
                   activation='relu'),
            MaxPooling2D(pool_size=hp.Choice('pool2_size', values=[1, 5], default=5)),
            Conv2D(filters=hp.Choice('conv3_filters', values=[128, 256], default=128),
                   kernel_size=hp.Choice('conv3_kernel_size', values=[1, 5], default=5),
                   activation='relu'),
            MaxPooling2D(pool_size=hp.Choice('pool3_size', values=[1, 5], default=5)),
            Flatten(),
            Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32), activation='relu',
                  kernel_regularizer=l1_l2(l1=hp.Float('l1_reg1', 1e-5, 1e-3, sampling='LOG', default=1e-4),
                                           l2=hp.Float('l2_reg1', 1e-5, 1e-3, sampling='LOG', default=1e-4))),
            Dense(units=hp.Int('units2', min_value=32, max_value=256, step=32), activation='relu',
                  kernel_regularizer=l1_l2(l1=hp.Float('l1_reg2', 1e-5, 1e-3, sampling='LOG', default=1e-4),
                                           l2=hp.Float('l2_reg2', 1e-5, 1e-3, sampling='LOG', default=1e-4))),
            Dense(units=hp.Int('units3', min_value=32, max_value=128, step=32), activation='relu',
                  kernel_regularizer=l1_l2(l1=hp.Float('l1_reg3', 1e-5, 1e-3, sampling='LOG', default=1e-4),
                                           l2=hp.Float('l2_reg3', 1e-5, 1e-3, sampling='LOG', default=1e-4))),
            Dense(units=hp.Int('units4', min_value=32, max_value=64, step=32), activation='relu',
                  kernel_regularizer=l1_l2(l1=hp.Float('l1_reg4', 1e-5, 1e-3, sampling='LOG', default=1e-4),
                                           l2=hp.Float('l2_reg4', 1e-5, 1e-3, sampling='LOG', default=1e-4))),
            Dense(25, activation=None),
            Reshape((5, 5)),
            Lambda(normalize_probability_distributions)
        ])

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(loss='categorical_cross_entropy',
                      optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      metrics=['accuracy'])
        return model



tuner2 = Hyperband(CNNHyperModelAll((512,512,3),5),
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt',
                      overwrite=True)


batch_size = 32
train_generator = dataset_generator(batch_size, sample_space_size=1000)
validation_generator = dataset_generator(batch_size, sample_space_size=200)

tuner2.search(train_generator, epochs=50, steps_per_epoch=1000 // batch_size, validation_data=validation_generator, validation_steps=200 // batch_size)

best_hps=tuner2.get_best_hyperparameters(num_trials=1)[0]

