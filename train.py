import os

import cv2

from albumentations import (
    CLAHE,
    Compose,
    HorizontalFlip,
    HueSaturationValue,
    JpegCompression,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    Rotate,
    ShiftScaleRotate,
    ToFloat,
)
from data_generator import Sentinel2MSIDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # for RTX GPUs


def simple_cnn():
    filter_dims = [32, 64, 128]
    height, width, channel = (240, 320, 13)
    num_of_classes = 10

    inputs = Input((height, width, channel,))

    _inputs = inputs

    for filter_dim in filter_dims:
        conv = Conv2D(
            filter_dim,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="glorot_uniform",
        )(_inputs)
        conv = Conv2D(
            filter_dim,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="glorot_uniform",
        )(conv)

        _inputs = MaxPool2D(pool_size=(2, 2))(conv)

    drop = Dropout(0.5)(_inputs)
    gap = GlobalAveragePooling2D()(drop)
    outputs = Dense(num_of_classes, activation="softmax")(gap)

    model = Model(inputs=inputs, outputs=outputs, name="SimpleCNN")

    return model


def get_resnet50():
    height, width, channel = (240, 320, 13)
    num_of_classes = 10

    base_model = ResNet50(
        input_shape=(height, width, channel),
        weights="imagenet",
        include_top=False,
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(units=num_of_classes, activation="softmax")(x)
    model = Model(
        inputs=base_model.input, outputs=predictions, name="ResNet50"
    )

    model.summary()

    return model


if __name__ == "__main__":
    # model = get_resnet50()
    model = simple_cnn()
    model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    AUGMENTATION_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
            ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            ToFloat(max_value=255),
        ]
    )

    data_path = "/home/notha99y/Workspace/satellite/sentinel_2/tif/"
    train_gen = Sentinel2MSIDataGenerator(
        data_path=data_path,
        batch_size=8,
        augmentations=AUGMENTATION_TRAIN,
        shuffle=True,
    )

    model.fit(
        train_gen,
        epochs=25,
        steps_per_epoch=train_gen.n // train_gen.batch_size + 1,
    )
