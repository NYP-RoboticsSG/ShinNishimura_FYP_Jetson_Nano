from dataclasses import dataclass
import numpy as np
import cv2
from configs import *

def pixel2dist(servo, pixel):
    y1 = -0.002526*servo**4 - 0.36426*servo**3 - 19.32845*servo**2 - 444.7515*servo - 3834.52
    y2 =  0.0157327*servo**4 + 2.2531533*servo**3 + 118.7390833*servo**2 + 2719.8811667*servo + 23497.55
    return np.exp((pixel-y2)/y1)

def dist2pixel(servo, dist):
    y1 = -0.002526*servo**4 - 0.36426*servo**3 - 19.32845*servo**2 - 444.7515*servo - 3834.52
    y2 =  0.0157327*servo**4 + 2.2531533*servo**3 + 118.7390833*servo**2 + 2719.8811667*servo + 23497.55
    return y1*np.log(dist)+y2

def mod_dist(img:np.ndarray, servo:int, d1=D1, d2=D2):
    line1, line2 = dist2pixel(servo, d1), dist2pixel(servo, d2)
    padsize, white = 600, (255, 255, 255)
    img = np.pad(img, ((padsize, padsize), (0, 0), (0, 0)), 'edge')
    img = img[padsize + int(line2): padsize + int(line1)]
    img = cv2.resize(img, (224, 224))
    return img

def pad(image:np.ndarray, pad_size=1, pad_value=255.0):
    under_image = np.full([shape + pad_size*2 for shape in image.shape], fill_value=pad_value)
    under_image[pad_size:image.shape[0]+pad_size, pad_size:image.shape[1]+pad_size] = image
    return under_image

def row_show(images:np.ndarray, image_height=100, max_image=8, pad_value=255.0, layer_name=''):
    for n in range(0, len(images), max_image):
        row_images = images[n:n+max_image]
        if len(range(0, len(images), max_image)) != 1:
            while len(row_images) < max_image:
                if len(images.shape) == 1:
                    row_images = np.concatenate((row_images, np.array([pad_value], dtype=np.float32)))
                else:
                    row_images = np.concatenate((row_images, np.full((1,) + images[0].shape, fill_value=pad_value, dtype=np.float32)), axis=0)
        for new_image in row_images:
            if len(images.shape) == 1:
                new_image = np.full((image_height, image_height), fill_value=new_image, dtype=np.float32)
            else:
                new_image = cv2.resize(new_image, (int(new_image.shape[1]/new_image.shape[0]*image_height), image_height), interpolation=0)
            _, new_image = cv2.threshold(new_image, 255.0, 0.0, cv2.THRESH_TRUNC)
            new_image = pad(new_image)
            try:
                image = np.concatenate((image, new_image), axis=0)
            except NameError: image = new_image
        try:
            output_image = np.concatenate((output_image, image), axis=1)
        except NameError: output_image = image
        del image
    output_image = pad(output_image)
    output_image = output_image.astype(np.uint8)
    return output_image

def unison_shuffled_copies(*args:np.ndarray) -> tuple([np.ndarray]):
    for arg in args:
        assert len(args[0]) == len(arg)
    p = np.random.permutation(len(args[0]))
    return tuple(arg[p] for arg in args)

def loading_bar(done, left, done_i='=', left_i='-', fold=LOADING_BAR_FOLD, insert=']\n['):
    s = done_i*done + left_i*left
    return (''.join(l + insert * (n % fold == fold-1) for n, l in enumerate(s)))

def train_stats(_current_epoch, _total_epoch,
                _current_batch, _total_batch,
                **kwargs):
    print(f'Epoch: {int(_current_epoch)+1}/{_total_epoch}')
    print(f'Batch: {int(_current_batch)}/{_total_batch}')
    print(f'[{loading_bar(int(_current_batch), int(_total_batch)-int(_current_batch))}]')
    if len(kwargs) > 0:
        maxlen = max(len(kwarg) for kwarg in kwargs)
        for kwarg in kwargs:
            print(f"{kwarg}: {(maxlen-len(kwarg))*' '}{kwargs[kwarg]}")
    print()

def aug_view(img, deg=0, shift=0):
    assert deg   in range(-56, 56+1)
    assert shift in range(-28, 28+1)
    h, w, c = img.shape
    origin = np.float32(
        [
            [int(w/4)+deg,   0                    ], [int(w/4*3)+deg,   0                    ],
            [int(w/8)+shift, h-int(h/8)-int(deg/2)], [w-int(w/8)+shift, h-int(h/8)+int(deg/2)]
        ]
    )
    new = np.float32(
        [[0, 0], [224, 0], [0, 224], [224, 224]]
    )
    M = cv2.getPerspectiveTransform(origin, new)
    img = cv2.warpPerspective(img, M, (224, 224))
    return img

class Stacker:
    @dataclass(frozen=True)
    class Stack:
        frame: float
        screen: np.ndarray

    def __init__(self, amount:int, interval:int):
        self.targets = tuple(n*interval for n in range(amount))
        self.frames = []

    def stack_frame(self, frame:float, screen:np.ndarray):
        self.frames.append(self.Stack(frame=frame, screen=screen))

    def get_frame(self, float:int) -> Stack:
        closeness = tuple(map((lambda stack: abs(frame-stack.frame)), self.frames))
        close_index = closeness.index(min(closeness))
        best_fit = self.frames[close_index]
        return best_fit

    def get_frames(self, current:float):
        best_fit = tuple(map(self.get_frame, (current-n for n in self.targets)))
        threshold = min(tuple(map((lambda stack: stack.frame), best_fit)))
        self.frames = list(filter((lambda stack: stack.frame >= threshold), self.frames))
        output = np.concatenate(tuple(map((lambda stack: stack.screen), best_fit)), axis=-1)
        return output

    def __len__(self):
        return len(self.frames)
