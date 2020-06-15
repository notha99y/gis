import collections
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from osgeo import gdal
from tensorflow.keras.utils import Sequence, to_categorical


class Sentinel2MSIDataGenerator(Sequence):
    def __init__(self, data_path, batch_size, augmentations, shuffle):

        self.data_path = data_path
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle

        self.X, self.y = self._get_X_and_y(self.data_path)
        self._on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Get batch of X and y"""

        _indexes = self.indexes[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        _batch_X = [self.X[k] for k in _indexes]
        _batch_y = [self.y[k] for k in _indexes]
        _batch_y_idx = [self.reverse_class_idx[k] for k in _batch_y]

        def read_image(image_path):
            """Reads a Sentinel-2's MSI path
            """
            raster = gdal.Open(str(image_path))
            rasterArray12 = raster.ReadAsArray()
            rasterArray8 = (rasterArray12 / 16).astype("uint8")
            return np.transpose(rasterArray8, axes=[2, 1, 0])

        batch_X = np.stack(
            [
                self.augmentations(image=read_image(X))["image"]
                for X in _batch_X
            ],
            axis=0,
        )
        batch_y = to_categorical(_batch_y_idx, len(self.class_names))

        return batch_X, batch_y

    def _on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _get_X_and_y(self, data_path):

        print("Reading from ", data_path)
        image_path = Path(data_path)
        assert image_path.is_dir(), "Data path given is not a directory!"
        _x = []
        _y = []
        for p in tqdm(image_path.rglob("*.tif")):
            _x.append(p)
            _y.append(p.parent.stem)

        self.class_names = sorted(list(set(_y)))
        _class_map = dict()
        _reverse_class_map = dict()
        for i, name in enumerate(self.class_names):
            _class_map[i] = name
            _reverse_class_map[name] = i
        self.class_idx = _class_map
        self.reverse_class_idx = _reverse_class_map

        self.n = len(_x)
        cnt = collections.Counter(_y)
        print(cnt)

        return _x, _y


if __name__ == "__main__":
    from albumentations import (
        Compose,
        HorizontalFlip,
        Rotate,
        CLAHE,
        HueSaturationValue,
        RandomBrightness,
        RandomContrast,
        RandomGamma,
        JpegCompression,
        ToFloat,
        ShiftScaleRotate,
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
