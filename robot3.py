from pyniryo import *
import time
from pyniryo import vision
import cv2
from skimage.color import rgb2gray
import numpy as np
from math import pi
from skimage.measure import label, regionprops,regionprops_table

robot = NiryoRobot('169.254.200.200')



def calibrate():
    robot.calibrate_auto()



def move_object():
    #robot.release_with_tool()
    print("first")
    post1 = robot.get_pose()
    print("second")
    robot.wait(10)
    post2 = robot.get_pose()
    print("moving")
    robot.pick_and_place(post1, post2)

def detect_and_move_object():
    robot.clear_collision_detected()
    obj_found, shape_ret, color_ret = robot.vision_pick("Tobor",
    height_offset = 0.0,
    shape = ObjectShape.SQUARE,
    color = ObjectColor.RED)

    cur = robot.get_pose()
    cur[0] = cur[0] + 0.1
    robot.move(cur)



detect_and_move_object()