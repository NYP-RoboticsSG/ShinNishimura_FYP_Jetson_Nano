import pandas as pd
import numpy as np
import cv2

from utils import *
from configs import *

import datetime
import random

class Dataset(object):
    def __init__(self, label_file=DATA_FILE,
                 batch_size=BATCH_SIZE, tts=TTS, seed=SEED):
        X, y = self.load_data(label_file, seed=seed)

        if SAMPLE_TYPE == 'roll':
            randroll = np.random.randint(0, y.shape[0])
            X, y = np.roll(X, randroll, axis=0), np.roll(y, randroll, axis=0)
        if SAMPLE_TYPE == 'shuffle':
            X, y = unison_shuffled_copies(X, y)

        test_datasets = round(y.shape[0]*tts)
        test_X,  test_y =  X[:test_datasets], y[:test_datasets]
        train_X, train_y = X[test_datasets:], y[test_datasets:]

        self.test_ds  = SubDataset(test_X,  test_y,  test=True,  batch_size=batch_size)
        self.train_ds = SubDataset(train_X, train_y, test=False, batch_size=batch_size)

    def load_data(self, filename:str, seed):
        df = pd.read_csv(filename, sep=' ', header=None)
        df.columns = [
            'imdir', 'left', 'right'
        ]
        df['frame'] = df['imdir'].apply(lambda s: s.split('/')[1].split('.')[0].split('_'))
        df['frame'] = df['frame'].apply(lambda t: (int(t[0])*60**2) + (int(t[1])*60) + int(t[2]) + (int(t[3])/1000000))
        X_df = pd.DataFrame()
        y_df = df[['left', 'right']]
        y_df = y_df.rolling(window=SHIFT*2+1, min_periods=1, center=True).mean()
        np.set_printoptions(threshold=np.inf)

        for n in range(STACKS):
            X_df[f'target_imdir_{n}'] = df['frame'] - INTERVAL*n
            X_df[f'target_imdir_{n}'] = X_df[f'target_imdir_{n}'].apply(
                lambda x: df.iloc[(df['frame']-x).abs().argsort()[:1]]['imdir'].tolist()[0]
            )

        return X_df.to_numpy()[30:600], y_df.to_numpy()[30:600]

class SubDataset(object):
    def __init__(self, X, y, test, batch_size):
        self.num_samples = y.shape[0]
        self.batch_size  = batch_size
        self.num_batches = int(np.ceil(y.shape[0]/self.batch_size))
        self.test = test

        self.X, self.y = X, y

    def load_img(self, path:str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img

    def load_imgs(self, paths:np.ndarray) -> np.ndarray:
        imgs = np.concatenate([self.load_img(path) for path in paths], -1)
        return imgs

    def augumentate(self, image, y):
        image = self.load_imgs(image)
        if self.test:
            image = aug_view(image)
            return image, y

        # random flip
        if random.random() < 0.5:
            image = image[:, ::-1, :]
            y = np.flip(y)

        # random rot and shift
        if random.random() < 0.5:
            deg   = random.randrange(-56, 56+1)
            shift = random.randrange(-28, 28+1)
            image[..., 0:3] = aug_view(image[..., 0:3], deg=deg, shift=shift)
            bonus = deg/56 + shift/56

            y[0] += (1.0 - abs(y[0]))*bonus*0.5 + bonus*0.5
            y[1] -= (1.0 - abs(y[1]))*bonus*0.5 + bonus*0.5
            y = np.clip(y, -1.0, 1.0)
        else: image[..., 0:3] = aug_view(image[..., 0:3])

        # random brightness
        if random.random() < 0.5:
            image = image.astype(np.int16)
            image[:, :, 2::3] += random.randint(-10, 10)
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
        return image, y

    def __iter__(self):
        self.batch_count = 0
        self.X, self.y = unison_shuffled_copies(self.X, self.y)
        return self

    def __next__(self):
        if self.batch_count < self.num_batches:
            X = self.X[self.batch_size*self.batch_count:self.batch_size*(self.batch_count+1)].copy()
            y = self.y[self.batch_size*self.batch_count:self.batch_size*(self.batch_count+1)].copy()

            Xy = tuple(zip(*(self.augumentate(ex, y) for ex, y in zip(X, y))))
            X, y = Xy

            X = np.stack(X).astype(np.float32)/255
            y = np.stack(y).astype(np.float32)

            self.batch_count += 1
            return X, y
        else:
            raise StopIteration


if __name__ == '__main__':
    ds = Dataset()
    for X, y in ds.train_ds:
        print(X.shape, y.shape)