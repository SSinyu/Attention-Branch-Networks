import numpy as np
from math import ceil
from scipy import ndimage
from sklearn.model_selection import train_test_split

from tensorflow.keras import utils
from tensorflow.keras.datasets.cifar10 import load_data


class DataLoader(utils.Sequence):
    def __init__(
            self,
            mode="train",
            batch_size=8,
            pad_size=None,
            crop_size=None,
            **kwargs):
        super(DataLoader, self).__init__(**kwargs)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_data()

        if mode in ["train", "valid"]:
            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
                self.x_train, self.y_train, test_size=0.2, random_state=2020
            )

        self.mode = mode
        self.batch_size = batch_size
        self.pad_size = pad_size
        self.crop_size = crop_size

        if self.crop_size is None:
            self.crop_size = self.x_train[0].shape[0]

        if mode == "train":
            self.indexes = np.arange(len(self.y_train))
        elif mode == "valid":
            self.indexes = np.arange(len(self.y_valid))
        else:
            self.indexes = np.arange(len(self.y_test))
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __len__(self):
        return ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]
        if self.mode == "train":
            batch_x = [random_rotate(random_flip(random_crop(pad(img, self.pad_size), self.crop_size))) for img in self.x_train[indexes]]
            batch_x = norm(np.array(batch_x)).astype(np.float32)
            batch_y = self.y_train[indexes]
        elif self.mode == "valid":
            batch_x = [img for img in self.x_valid[indexes]]
            batch_x = norm(np.array(batch_x)).astype(np.float32)
            batch_y = self.y_valid[indexes]
        else:
            batch_x = [img for img in self.x_test[indexes]]
            batch_x = norm(np.array(batch_x)).astype(np.float32)
            batch_y = self.y_test[indexes]

        return batch_x, [batch_y, batch_y]


def norm(x):
    return (x - x.min()) / (x.max() - x.min())


def pad(x, pad_size):
    pad_width = [(pad_size, pad_size) for _ in range(2)] + [(0,0)]
    return np.pad(x, pad_width, "constant")


def random_crop(x, crop_size):
    h, w, _ = x.shape
    if (h > crop_size) and (w > crop_size):
        d1 = np.random.randint(0, (h-crop_size))
        d2 = np.random.randint(0, (w-crop_size))
        return x[d1:(d1+crop_size), d2:(d2+crop_size), :]
    else:
        return x


def random_flip(x):
    if np.random.random() < .5:
        x = x[::-1, ...]
    if np.random.random() < .5:
        x = x[:, ::-1, :]
    return x


def random_rotate(x):
    if np.random.random() < .5:
        angle = np.random.randint(*(0,360))
        x = ndimage.rotate(x, angle, reshape=False, mode="constant")
    return x


def get_generator(args):
    params = {
        "mode":args.mode,
        "batch_size":args.batch_size,
        "pad_size":args.pad_size,
        "crop_size":args.crop_size
    }
    return DataLoader(**params)
