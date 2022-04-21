from tensorflow import keras


class CustomCallback(keras.callbacks.Callback):
    # https://keras.io/guides/writing_your_own_callbacks/

    def on_train_begin(self, logs=None):
        self.weights_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') == 1:
            self.model.stop_training = True
        self.weights_accuracy.append([logs.get('accuracy'), self.model.get_weights()])

    def on_train_end(self, logs=None):
        self.weights_accuracy.sort(key=lambda x: x[0])
        self.model.set_weights(self.weights_accuracy[-1][1])


class PlainNetworkGenerator:

    callback = CustomCallback()

    def create_q_model(self, state_size, action_size, channel):
        observation = keras.layers.Input(shape=state_size[0] * state_size[1] * channel, name='input')
        layer1 = keras.layers.Dense(64, activation="relu")(observation)
        layer2 = keras.layers.Dense(64, activation="relu")(layer1)
        action = keras.layers.Dense(action_size, activation="linear")(layer2)

        model = keras.Model(inputs=observation, outputs=action)
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy'],
        )

        model.summary()
        return model


class ConvolutionNetworkGenerator:

    callback = CustomCallback()

    def create_q_model(self, state_size, action_size, channel):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(256, (3, 3), input_shape=(state_size[0], state_size[1], channel)))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Conv2D(256, (3, 3)))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64))

        model.add(keras.layers.Dense(action_size, activation="linear"))

        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy'],
        )

        model.summary()
        return model


class ConvolutionAtariNetworkGenerator:

    callback = CustomCallback()

    def create_q_model(self, state_size, action_size, channel):
        # Network defined by the Deepmind paper
        inputs = keras.layers.Input(shape=(state_size[0], state_size[1], channel))

        # Convolutions on the frames on the screen
        layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = keras.layers.Flatten()(layer3)

        layer5 = keras.layers.Dense(512, activation="relu")(layer4)
        action = keras.layers.Dense(action_size, activation="linear")(layer5)

        model = keras.Model(inputs=inputs, outputs=action)
        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        model.compile(
            loss="mse",
            optimizer=optimizer,
            metrics=['accuracy'],
        )

        model.summary()
        return model
