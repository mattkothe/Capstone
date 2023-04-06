from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Softmax
from Dataset import dataset_generator


def create_circle_detection_model_2(num_circles=5):
    inputs = Input(shape=(512, 512, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_circles * 5, activation='linear')(x)
    outputs = Reshape((num_circles, 5))(outputs)
    outputs = Softmax()(outputs)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with appropriate loss function and optimizer
    model.compile(loss='categorical_cross_entropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


train_dataset = dataset_generator(100)

model = create_circle_detection_model_2()

sample_size = 100
batch_size = 8

model.fit(train_dataset,epochs=50,steps_per_epoch=sample_size/batch_size,batch_size=batch_size)


