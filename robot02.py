from pyniryo import *
import time
from pyniryo import vision
import cv2
from skimage.color import rgb2gray
import numpy as np
from math import pi
from skimage.measure import label, regionprops,regionprops_table

robot = NiryoRobot('169.254.200.200')



def calibrate(robot):
    robot.calibrate_auto()


def get_image(robot):
    x1 = robot.get_joints()
    x = x1
    x[0] = x[0] - 1.6
    x[2] = x[2] + 1
    x[4] = x[4] - 1.7
    robot.move(x)
    img_c = robot.get_img_compressed()
    img = vision.uncompress_image(img_c)
    cv2.imwrite("outputSHOW.jpg", img)
    robot.move_to_home_pose()

def get_image_workspace():
    x1 = robot.get_joints()
    x = x1
    x[0] = x[0] - 1.6
    x[2] = x[2] + 1
    x[4] = x[4] - 1.7
    robot.move(x)
    mtx, dist = robot.get_camera_intrinsics()
    img_c = robot.get_img_compressed()
    img = vision.uncompress_image(img_c)
    img = vision.undistort_image(img, mtx, dist)
    img = vision.extract_img_workspace(img, 1)
    #img = cv2.resize(img, (640, 480))
    cv2.imwrite("outputGREENWORKSPACE.jpg", img)
    robot.move_to_home_pose()

def detection():
    print(robot.detect_object("Tobor", ObjectShape.ANY, ObjectColor.RED))
    print(robot.get_target_pose_from_cam("Tobor", 1,ObjectShape.ANY, ObjectColor.RED))

    robot.clear_collision_detected()
    robot.move_to_object("Tobor", 0.1,ObjectShape.ANY, ObjectColor.RED)


def img_stream():
    x1 = robot.get_joints()
    x = x1
    x[0] = x[0] - 1.6
    x[2] = x[2] + 1
    x[4] = x[4] - 1.7
    robot.move(x)
    mtx, dist = robot.get_camera_intrinsics()

    while True:
        img_c = robot.get_img_compressed()
        img = vision.uncompress_image(img_c)
        img = vision.undistort_image(img, mtx, dist)
        img = vision.extract_img_workspace(img, 1)
        #img = cv2.resize(img, (640, 480))
        key = vision.show_img("window",img,1)

        if key in [27, ord("q")]:  #
            robot.move_to_home_pose()
            break




from pyniryo import NiryoRobot, PoseObject
from pyniryo import vision
from pyniryo.vision.image_functions import ColorHSV, ObjectType
import cv2

import cv2
import numpy as np

def detect_and_move_own(robot,
                        workspace_name="Tobor",
                        workspace_id=1,
                        color_hsv=ColorHSV.GREEN,
                        shape=ObjectType.SQUARE,
                        approach_z=0.12,
                        touch_z=0.03,
                        do_touch=False,
                        close_gripper=False):
    x1 = robot.get_joints()
    x = x1
    x[0] = x[0] - 1.6
    x[2] = x[2] + 1
    x[4] = x[4] - 1.7
    robot.move(x)
    mtx, dist = robot.get_camera_intrinsics()
    # 1) Capture and prep image
    img_c = robot.get_img_compressed()
    img = vision.uncompress_image(img_c)
    img = vision.undistort_image(img, mtx, dist)
    ws_img = vision.extract_img_workspace(img, workspace_id)

    # 2) Threshold using built-in helper
    mask = vision.threshold_hsv(ws_img, *color_hsv.value)

    # 3) Detect contour
    cnt = vision.biggest_contour_finder(mask)
    if cnt is None or len(cnt) == 0:
        print(f"[INFO] No object found for color={color_hsv}")
        return

    # --- shape decision (simple) ---
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (peri * peri + 1e-6)

    detected_shape = "CIRCLE" if circularity > 0.85 else "SQUARE"
    print(f"[OK] Detected shape: {detected_shape} (circularity={circularity:.3f})")

    # optional: enforce desired shape
    if (shape == ObjectType.SQUARE and detected_shape != "SQUARE") or \
       (shape == ObjectType.CIRCLE and detected_shape != "CIRCLE"):
        print(f"[INFO] Object shape does not match required {shape}, skipping.")
        return

    # 4) Center and yaw
    cx, cy = vision.get_contour_barycenter(cnt)
    angle_rad = vision.get_contour_angle(cnt)
    print(f"[OK] Found object at px=({cx:.1f}, {cy:.1f}), yaw={angle_rad:.3f} rad")

    # 5) Pixel → relative position (meters)
    rel_x, rel_y = vision.relative_pos_from_pixels(ws_img, int(round(cx)), int(round(cy)))
    print(f"[OK] Relative XY: ({rel_x:.4f}, {rel_y:.4f}) m")

    obj_pose = robot.get_target_pose_from_rel(
        workspace_name,
        height_offset=0.0,
        x_rel=rel_x,
        y_rel=rel_y,
        yaw_rel=angle_rad
    )

    print(robot.get_pose())
    print(obj_pose)
    robot.move(obj_pose)
    robot.pick(obj_pose)
    robot.move_to_home_pose()
    print("[DONE] Detection & move sequence complete.")






#calibrate(robot)
robot.move_to_home_pose()
#robot.move_to_home_pose()
#get_image(robot)
#detection()
#img_stream()
get_image_workspace()

'''detect_and_move_own(robot,
                            workspace_id=1,
                            color_hsv=ColorHSV.BLUE,
                            shape=ObjectType.SQUARE,
                            approach_z=0.15,
                            do_touch=True,
                            close_gripper=True)'''




