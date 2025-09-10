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


def get_image():
    img_c = robot.get_img_compressed()
    img = vision.uncompress_image(img_c)
    cv2.imwrite("output03.jpg", img)

def detection():
    print(robot.detect_object("Tobor", ObjectShape.ANY, ObjectColor.RED))
    print(robot.get_target_pose_from_cam("Tobor", 1,ObjectShape.ANY, ObjectColor.RED))

    robot.clear_collision_detected()
    robot.move_to_object("Tobor", 0.1,ObjectShape.ANY, ObjectColor.RED)


def img_stream():
    while True:
        img_c = robot.get_img_compressed()
        img = vision.uncompress_image(img_c)
        img = vision.extract_img_workspace(img, 1)
        img = cv2.resize(img, (640, 480))
        vision.show_img("window",img,1)


def detection_enhanced():
    img_c = robot.get_img_compressed()
    img = vision.uncompress_image(img_c)
    img = vision.extract_img_workspace(img, 1)
    img = rgb2gray(img) > 0.5
    #cv2.imwrite("outputTEST.jpg", img)

    print(img.shape)
    label_img = label(img)

    props = regionprops(label_img)

    for region in props:
        area = region.area
        perimeter = region.perimeter

        if perimeter == 0:
            circ = 0
        else:
            circ = 4 * np.pi * area / (perimeter ** 2)

        print(f"Label {region.label}: circularity={circ:.3f}")



def get_workspace_bgr(robot, workspace_id:int=1):
    img_c = robot.get_img_compressed()
    img = vision.uncompress_image(img_c)             # BGR
    img_ws = vision.extract_img_workspace(img, workspace_id)
    return img_ws

# --- HSV masky pro barvy (OpenCV: H∈[0,179]) ---
def color_mask_hsv_bgr(bgr, color: str):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if color == "red":
        # Červená
        m1 = cv2.inRange(hsv, (0,   90, 60), (8,   255, 255))
        m2 = cv2.inRange(hsv, (170, 90, 60), (179, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif color == "green":
        # Zelená ~ [35–95]
        mask = cv2.inRange(hsv, (35, 60, 50), (95, 255, 255))
    elif color == "blue":
        # Modrá ~ [95–130]
        mask = cv2.inRange(hsv, (95, 60, 50), (130, 255, 255))
    else:
        raise ValueError("color must be one of {'red','green','blue'}")

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    return mask


def circularity(contour):
    a = cv2.contourArea(contour)
    p = cv2.arcLength(contour, True)
    if p <= 0:
        return 0.0
    return 4.0 * np.pi * a / (p * p)

def is_square(contour, tol_aspect=0.25):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return False
    x, y, w, h = cv2.boundingRect(approx)
    if w == 0 or h == 0:
        return False
    aspect = (w/h) if w >= h else (h/w)
    return aspect <= (1.0 + tol_aspect)

def is_circle(contour, min_circ=0.90):
    return circularity(contour) >= min_circ

def contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (M["m10"]/M["m00"], M["m01"]/M["m00"])

def contour_yaw(contour):
    (_, _), (w, h), angle_deg = cv2.minAreaRect(contour)  # angle in (-90,0]
    angle_rad = np.deg2rad(angle_deg)
    if w < h:
        angle_rad += np.pi/2
    if angle_rad <= -np.pi/2:
        angle_rad += np.pi
    if angle_rad > np.pi/2:
        angle_rad -= np.pi
    return float(angle_rad)

# --- (cx, cy, yaw) or None ---
def find_object_by_color_and_shape_bgr(bgr, color="red", want_shape="square", min_area = 0):
    mask = color_mask_hsv_bgr(bgr, color)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = -1
    for c in cnts:
        area = cv2.contourArea(c)


        ok = (is_square(c) if want_shape == "square" else is_circle(c))
        if not ok:
            continue

        center = contour_center(c)
        if center is None:
            continue

        if area > best_area:
            cx, cy = center
            yaw = 0.0 if want_shape == "circle" else contour_yaw(c)
            best = (cx, cy, yaw)
            best_area = area

    return best


def move_over_detected(robot,
                       workspace_id=1,
                       workspace_name="Tobor",
                       color="red",
                       want_shape="square",
                       min_area=0,
                       approach_z=0.2,   # bezpečná výška nad stolem
                       touch_z=0.03,      # výška pro "dosed" (pokud chceš)
                       do_touch=False,    # pokud True, sjede na touch_z
                       close_gripper=False):
    """
    1) Detekuje objekt (barva+tvar).
    2) Přepočte pixely -> relativní (x,y) v metrech pomocí vision.relative_pos_from_pixels(img, x, y).
    3) Provede jednoduchou trajektorii:
        - zvednout do approach_z,
        - přesun XY nad cíl, nastavit yaw,
        - volitelně sjet na touch_z (a případně zavřít gripper).
    """

    # 1) snímek workspace (BGR)
    bgr = get_workspace_bgr(robot, workspace_id=workspace_id)

    # 2) detekce
    res = find_object_by_color_and_shape_bgr(bgr, color=color, want_shape=want_shape, min_area=min_area)
    if res is None:
        print(f"[INFO] not Found {color} {want_shape}..")
        return
    cx, cy, yaw = res
    print(f"[OK] Found {color} {want_shape}: px=({cx:.1f},{cy:.1f}), yaw={yaw:.3f} rad")

    # 3) pix->rel XY (m)
    rel_x, rel_y = vision.relative_pos_from_pixels(bgr, int(round(cx)), int(round(cy)))
    rel_yaw = float(yaw)
    print(f"[OK] Relatiive XY: ({rel_x:.4f}, {rel_y:.4f}) m; yaw={rel_yaw:.3f} rad")

    # 4) aktuální póza (z ní uděláme relativní posun)
    cur = robot.get_pose()
    print(cur)

    cur[0] = cur[0] + 0.1
    cur[1] = cur[1] + 0.1
    cur[5] = cur[5] + 0.1
    print(cur)
    robot.move(cur)

    print("[DONE]")




#robot.move_to_home_pose()
#get_image()
#detection()
#img_stream()

#detection_enhanced()
# Jednoduchý přesun nad červený čtverec
move_over_detected(robot,
                   workspace_id=1,
                   workspace_name="Tobor",
                   color="green",
                   want_shape="square",
                   approach_z=0.13,
                   do_touch=False)

