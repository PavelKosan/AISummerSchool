from pyniryo import *
import time
from pyniryo import vision
import cv2
from skimage.color import rgb2gray
import numpy as np
from math import pi
from skimage.measure import label, regionprops,regionprops_table

robot = NiryoRobot('169.254.200.200')


def quality_control_process(robot,
                            workspace_name="Test02",
                            workspace_id=1,
                            good_shape=ObjectShape.SQUARE):
    conveyor_id = robot.set_conveyor()

    x = robot.get_joints()
    x[1] = x[1] + 0.2
    x[2] = x[2] + 0.5
    x[4] = x[4] - 1.5
    robot.move(x)

    print("[INFO] Starting QC process...")
    robot.run_conveyor(conveyor_id,20,ConveyorDirection.FORWARD)

    # camera intrinsics
    mtx, dist = robot.get_camera_intrinsics()

    while True:
        # --- Capture image ---
        img_c = robot.get_img_compressed()
        img = vision.uncompress_image(img_c)
        img = vision.undistort_image(img, mtx, dist)
        ws_img = vision.extract_img_workspace(img, workspace_id)

        # --- Find contour (for shape decision only) ---
        gray = cv2.cvtColor(ws_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        cnt = vision.biggest_contour_finder(mask)

        if cnt is None or len(cnt) == 0:
            continue

        # Shape decision
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (peri * peri + 1e-6)
        detected_shape = ObjectShape.CIRCLE if circularity > 0.85 else ObjectShape.SQUARE
        print(f"[QC] Detected shape: {detected_shape}, circ={circularity:.3f}")

        # --- If bad product ---
        if detected_shape != good_shape:
            print("[QC] Bad product → stopping conveyor")
            robot.stop_conveyor(conveyor_id)

            # Use built-in vision_pick (no manual pose)
            obj_found, shape_ret, color_ret = robot.vision_pick(
                workspace_name,
                height_offset=0.0,
                shape=detected_shape,
                color=ObjectColor.ANY
            )
            if obj_found:
                print(f"[QC] Removed bad product ({detected_shape})")
                robot.clear_collision_detected()

                cur = robot.get_joints()
                cur[0] = - 1.6
                robot.move(cur)
                cur = robot.get_joints()
                cur[1] = cur[1] - 0.1
                robot.move(cur)
                robot.release_with_tool()
                cur = robot.get_joints()
                cur[1] = cur[1] + 0.3
                robot.move(cur)
                robot.move(x)
            else:
                print("[QC] vision_pick could not grab object")


            # Restart conveyor
            robot.run_conveyor(conveyor_id, ConveyorDirection.FORWARD, 300)
            print("[QC] Conveyor restarted")

        # --- Exit key ---
        key = vision.show_img("QC Window", ws_img, 1)
        if key in [27, ord("q")]:
            break

    # cleanup
    robot.stop_conveyor(conveyor_id)
    robot.move_to_home_pose()
    print("[INFO] QC process finished")



'''quality_control_process(robot,
                            workspace_name="Test02",
                            workspace_id=1,
                            good_shape=ObjectShape.SQUARE)'''

conveyor_id = robot.set_conveyor()
robot.stop_conveyor(conveyor_id)
