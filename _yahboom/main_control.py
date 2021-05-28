from _servoserial import ServoSerial

class Servo:
    def __init__(self, yaw=0.0, pitch=0.0):
        self.device = ServoSerial()
        
        self.home_pos = yaw, pitch
        self.yaw, self.pitch = yaw, pitch
        self.goto(self.yaw, self.pitch)
        
    def home(self):
        self.goto(*self.home_pos)
        
    def goto(self, yaw=None, pitch=None, coding='deg'):

        if yaw is None:   
            yaw = self.yaw
        else:
            self.yaw = yaw
        if pitch is None: 
            pitch = self.pitch
        else:
            self.pitch = pitch
            
        if coding == 'deg':
            yaw, pitch = self.to_serial(yaw, pitch)
        self.device.Servo_serial_double_control(1, yaw, 2, pitch)
        
    def to_degrees(self, yaw=None, pitch=None) -> tuple:
        if yaw is not None:
            yaw = (yaw-600)/2800*180-90
        if pitch is not None:
            pitch = (pitch-1300)/2795*180-45
        return yaw, pitch
    
    def to_serial(self, yaw=None, pitch=None) -> tuple:
        if yaw is not None:
            yaw = (yaw+90)/180*2800+600
        if pitch is not None:
            pitch = (pitch+45)/180*2795+1300
        return (None if yaw is None else round(yaw)), (None if yaw is None else round(pitch))
    
'''
CAMERA WITH 
'''
from jetbot import Camera as ConfusingCamera
from jetbot import bgr8_to_jpeg
from datetime import datetime
import numpy as np
import cv2

import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg
import traitlets
class Camera:
    def __init__(self, width=224, height=224, stream=True, transform=(lambda x: x)):
        self.device = ConfusingCamera.instance(width=width, height=height)
        self.w, self.h = width, height
        if stream:
            self.start_stream(transform)
        
    def stop_stream(self):
        del self.link
        
    def start_stream(self, transform=(lambda x: x)):
        def full_transform(x):
            x = transform(x)
            x = bgr8_to_jpeg(x)
            return x
        self.stream = widgets.Image(format='jpeg', width=self.w, height=self.h)
        self.link = traitlets.dlink((self.device, 'value'), (self.stream, 'value'), transform=full_transform)
        
        
    def get_image(self, dtype='array') -> np.ndarray:
        if dtype == 'array':
            return self.device.value
        
        if dtype == 'bytes':
            return bgr8_to_jpeg(self.device.value)

    def save_image(self, name=None) -> str:
        if name is None:
            name = str(datetime.now().time()).replace(':', '_') + '.jpg'
        else: 
            name = name.format(str(datetime.now().time()).replace(':', '_').replace('.', '_'))
        cv2.imwrite(name, self.get_image())
        return name

'''
ROBOT WITH SIMPLE STEERING
'''
from jetbot import Robot as BruhRobot
import time
class Robot:
    def __init__(self):
        self.device = BruhRobot()
        
    def movehead(self, amount=0.0, wait=0.0):
        self.device.up(amount)
        time.sleep(wait)
        self.device.vertical_motors_stop()
    
    '''
    steer left  = negative steer
    steer right = positive steer
    steering value range: -1.0 ~ 1.0
    '''
    def custom_steer(self, l_speed, r_speed, mult=1.0, 
            leftmax=0.95, leftmin=1.0, rightmax=1.0, rightmin=0.93):
        
        if l_speed > 0.0:
            l_speed = leftmax * abs(l_speed)**0.5
        if l_speed < 0.0:
            l_speed = leftmin * -abs(l_speed)**0.5
        
        if r_speed > 0.0:
            r_speed = leftmax * abs(r_speed)**0.5
        if r_speed < 0.0:
            r_speed = leftmin * -abs(r_speed)**0.5
            
        self.device.set_motors(l_speed*mult, r_speed*mult)
        
    
    def forward(self, steer=0.0, speed=1.0, left_weight=0.95, right_weight=1.0):
        left  = speed*left_weight
        right = speed*right_weight
        
        # steering right, slow down left
        if steer > 0.0:
            left = (1.0 - abs(steer)) * left
        # steering left,  slow down right
        if steer < 0.0:
            right = (1.0 - abs(steer)) * right
        
        self.device.set_motors(left, right)
        
    def backward(self, steer=0.0, speed=1.0, left_weight=1.0, right_weight=0.93):
        left  = -speed*left_weight
        right = -speed*right_weight
        
        # steering right, slow down right
        if steer > 0.0:
            right = (1.0 - abs(steer)) * right
        # steering left,  slow down left
        if steer < 0.0:
            left = (1.0 - abs(steer)) * left
            
        self.device.set_motors(left, right)
        
    def stop(self):
        self.device.stop()
        
'''
BATTERY LEVEL CHECKER
Initialize BatteryLevel and do .Update() to get current level
'''
from _Battery_Vol_Lib import BatteryLevel

'''
DISTANCE CALCULATION
'''
def pixel2dist(servo, pixel):
    y1 = -0.002526*servo**4 - 0.36426*servo**3 - 19.32845*servo**2 - 444.7515*servo - 3834.52
    y2 =  0.0157327*servo**4 + 2.2531533*servo**3 + 118.7390833*servo**2 + 2719.8811667*servo + 23497.55
    return np.exp((pixel-y2)/y1)

def dist2pixel(servo, dist):
    y1 = -0.002526*servo**4 - 0.36426*servo**3 - 19.32845*servo**2 - 444.7515*servo - 3834.52
    y2 =  0.0157327*servo**4 + 2.2531533*servo**3 + 118.7390833*servo**2 + 2719.8811667*servo + 23497.55
    return y1*np.log(dist)+y2
