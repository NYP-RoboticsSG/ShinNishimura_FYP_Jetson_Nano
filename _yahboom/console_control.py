
from main_control import Servo, Camera, Robot, BatteryLevel
import threading
import time

# controller = widgets.Controller(index=1)

'''
Sliders Range: -1.0 to 1.0
0: left joystick X
1: left joystick Y
2: right joystick Y
3: right joystick X
4: move pad X
5: move pad Y

Buttons Range: 0.0 or True
0: number 1 button
1: number 2 button
2: number 3 button
3: number 4 button
4: left top side button
5: right top side button
6: left bottom side button
7: right bottom side button
8: select button
9: start button
10: left joystick click
11: right joystick click
'''

import ipywidgets.widgets as widgets
_controller = widgets.Controller(index=0)

class Controller:
    def __init__(self, servo=Servo(), camera=Camera(), robot=Robot()):
        self.servo = servo
        self.camera = camera
        self.robot = robot
    
    def start_thread(self):
        self.thread = threading.Thread(target=self.mainloop)
        self.thread.start()
    
    def mainloop(self):
        global _controller
        print('started')
        print(_controller.axes)
        while True:
            if _controller.connected:
                if not _controller.buttons[3].value:
                    if (round(_controller.axes[1].value, 2) != 0.0) or (round(_controller.axes[2].value, 2) != 0.0):
                        self.robot.custom_steer(-_controller.axes[1].value, -_controller.axes[2].value, 0.7, 
                                               1.0, 1.0, 1.0, 1.0)

    #                 if _controller.axes[1].value < 0.0:
    #                     self.robot.forward(-_controller.axes[0].value, abs(_controller.axes[1].value)**0.5, 1.0, 1.0)
    #                 elif _controller.axes[1].value > 0.0:
    #                     self.robot.backward(_controller.axes[0].value, abs(_controller.axes[1].value)**0.5, 1.0, 1.0)

                    else:
                        self.robot.stop()
                    time.sleep(0.1)

    