import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    MaxPool3D,
    Activation,
    Reshape,
    SpatialDropout3D,
    BatchNormalization,
    TimeDistributed,
    Flatten,
)

# from tensorflow.keras.initializers import Orthogonal


def load_model() -> Sequential:
    model = Sequential()

    # First Conv3D Block
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    # Second Conv3D Block
    model.add(Conv3D(256, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    # Third Conv3D Block
    model.add(Conv3D(75, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    # Reshape before TimeDistributed
    # Original: (75, 46, 140) -> After MaxPool3D operations: (75, 5, 17)
    model.add(
        Reshape((75, -1))
    )  # Flatten spatial dimensions while keeping time dimension

    # LSTM layers
    model.add(
        Bidirectional(LSTM(128, kernel_initializer="orthogonal", return_sequences=True))
    )
    model.add(Dropout(0.5))

    model.add(
        Bidirectional(LSTM(128, kernel_initializer="orthogonal", return_sequences=True))
    )
    model.add(Dropout(0.5))

    # Output layer
    model.add(
        Dense(
            41,
            kernel_initializer="he_normal",
            activation="softmax",
        )
    )

    model.load_weights(os.path.join("model_3_4292", "checkpoint.weights.h5"))

    return model
