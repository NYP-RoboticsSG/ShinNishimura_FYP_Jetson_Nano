def append_txt_file(s:str, directory:str) -> str:
    with open(directory, 'a') as f:
        f.write(s)
    return s

def get_txt_file(directory:str) -> list:
    with open(directory, 'r') as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines if (line[-1] == '\n')]
    return lines

import numpy as np
import cv2
def aug_view(img, deg=0, shift=0):
    assert deg   in range(-56, 56+1)
    assert shift in range(-28, 28+1)
    h, w, c = img.shape
    origin = np.float32(
        [
            [int(w/4)+deg,   0                    ], [int(w/4*3)+deg,   0                    ],
            [int(w/8)+shift, h-int(h/8)-int(deg/2)], [w-int(w/8)+shift, h-int(h/8)+int(deg/2)]
#             [int(w/8)+shift, h                    ], [w-int(w/8)+shift, h                    ]
        ]
    )
    new = np.float32(
        [[0, 0], [224, 0], [0, 224], [224, 224]]
    )
    M = cv2.getPerspectiveTransform(origin, new)
    img = cv2.warpPerspective(img, M, (224, 224))
    return img

from dataclasses import dataclass
class Stacker:
    @dataclass(frozen=True)
    class Stack:
        frame: float
        screen: np.ndarray

    def __init__(self, amount:int, interval:float):
        self.targets = tuple(n*interval for n in range(amount))
        self.frames = []

    def stack_frame(self, frame:float, screen:np.ndarray):
        self.frames.append(self.Stack(frame=frame, screen=screen))

    def get_frame(self, frame:float) -> Stack:
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
    
def aug_road(img:np.ndarray, top_shift:int=0, bottom_shift:int=0) -> np.ndarray:
    assert top_shift in range(-32, 32+1)
    assert bottom_shift in range(-32, 32+1)
    h, w = img.shape[0], img.shape[1]
    old_point = np.float32([
        [0+32+top_shift, 0],    [w-32+top_shift, 0],
        [0-32-bottom_shift, h], [w+32-bottom_shift, h]
    ])
    new_point = np.float32([
        [0, 0],  [96, 0],
        [0, 96], [96, 96],
    ])
    M = cv2.getPerspectiveTransform(old_point, new_point)
    img = cv2.warpPerspective(img, M, (96, 96), borderMode=cv2.BORDER_REPLICATE)
    return img

import time
class FPS:
    class TimerNotStartedError(Exception): pass

    def __init__(self):
        self.started = False
        self.frames = 0

    def step(self):
        if not self.started:
            self.started = True
            self.start_time = time.perf_counter()
        self.frames += 1

    def get_fps(self, roundto=1):
        try:
            return round(self.frames/(time.perf_counter()-self.start_time))
        except AttributeError:
            raise self.TimerNotStartedError