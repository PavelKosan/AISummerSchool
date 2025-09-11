from pyniryo import *
import time
from pyniryo import vision
import cv2
from skimage.color import rgb2gray
import numpy as np
from math import pi
from skimage.measure import label, regionprops,regionprops_table

robot = NiryoRobot('169.254.200.200')
id = robot.set_conveyor()

def calibrate():
    robot.calibrate_auto()

def try_conveyor(id):
    robot.run_conveyor(id,20,ConveyorDirection.FORWARD)
    robot.wait(10)
    robot.run_conveyor(id,20,ConveyorDirection.BACKWARD)
    robot.wait(10)
    robot.stop_conveyor(id)


def stoping_conveyor(id):
    robot.run_conveyor(id, 20, ConveyorDirection.FORWARD)
    state = robot.digital_read(PinID.DI5)
    print(state)
    while state == PinState.HIGH:
        state = robot.digital_read(PinID.DI5)
        print(state)


    robot.stop_conveyor(id)

def get_it_to_conveyor():

    robot.run_conveyor(id, 20, ConveyorDirection.FORWARD)

    x_org = robot.get_joints()
    x = robot.get_joints()
    print(x_org)

    x[0] = x[0] - 1.6
    x[2] = x[2] + 1
    x[4] = x[4] - 1.7
    robot.move(x)

    robot.clear_collision_detected()

    obj_found, shape_ret, color_ret = robot.vision_pick("Tobor",
    height_offset = 0.0,
    shape = ObjectShape.SQUARE,
    color = ObjectColor.RED)

    robot.move_to_home_pose()
    cur = robot.get_joints()
    cur[1] = cur[1] - 0.1
    cur[2] = cur[2] + 0.1
    robot.move(cur)
    robot.release_with_tool()
    #robot.place(cur)
    state = robot.digital_read(PinID.DI5)
    print(state)
    while state == PinState.HIGH:
        state = robot.digital_read(PinID.DI5)
        print(state)

    robot.stop_conveyor(id)
    robot.move_to_home_pose()


def pick_and_place_from_conveyor():
    x_org = robot.get_joints()
    x = robot.get_joints()
    print(x_org)

    x[1] = x[1] + 0.2
    x[2] = x[2] + 0.5
    x[4] = x[4] - 1.5
    robot.move(x)

    mtx, dist = robot.get_camera_intrinsics()
    img_c = robot.get_img_compressed()
    img = vision.uncompress_image(img_c)
    img = vision.undistort_image(img, mtx, dist)
    img = vision.extract_img_workspace(img, 1)
    # img = cv2.resize(img, (640, 480))
    cv2.imwrite("belt2.jpg", img)

    robot.clear_collision_detected()

    obj_found, shape_ret, color_ret = robot.vision_pick("Test02",
                                                        height_offset=0.0,
                                                        shape=ObjectShape.SQUARE,
                                                        color=ObjectColor.BLUE)
    robot.clear_collision_detected()

    cur = robot.get_joints()
    cur[0] = - 1.6
    robot.move(cur)
    cur = robot.get_joints()
    cur[1] = cur[1] - 0.3
    robot.move(cur)
    robot.release_with_tool()
    cur = robot.get_joints()
    cur[1] = cur[1] + 0.3
    robot.move(cur)

    robot.move_to_home_pose()


robot.calibrate_auto()
robot.clear_collision_detected()
robot.move_to_home_pose()
#try_conveyor(id)
#stoping_conveyor(id)
#get_it_to_conveyor()
pick_and_place_from_conveyor()


