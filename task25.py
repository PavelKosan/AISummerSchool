import cv2
import numpy as np
from time import sleep
from pyniryo import *
from pyniryo import image_functions as vision


Beta_zone_drop_1 = PoseObject(-0.2309204655384558, 0.22429980658207935, 0.06, -2.505805460272864, 1.4967962544725275, 0.744181935125712)

# --- (Your detection functions: hsv_color_segmentation, find_and_locate_objects, draw_object_info) ---
# These are the same as the previous response.
def hsv_color_segmentation(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_red1 = np.array([0, 120, 70]); high_red1 = np.array([10, 255, 255])
    low_red2 = np.array([170, 120, 70]); high_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv_frame, low_red1, high_red1)
    red_mask2 = cv2.inRange(hsv_frame, low_red2, high_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    low_blue = np.array([94, 80, 50]); high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    low_green = np.array([40, 40, 40]); high_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    # cv2.imshow("Red Mask", red_mask)
    # cv2.imshow("Green Mask", green_mask)
    # cv2.imshow("Blue Mask", blue_mask)
    # cv2.waitKey(10000)
    return red_mask, green_mask, blue_mask

def color_segmentation_otsu(frame):
    red_frame = frame[:, :, 2]
    green_frame = frame[:, :, 1]
    blue_frame = frame[:, :, 0]
    _, red_mask = cv2.threshold(red_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, green_mask = cv2.threshold(green_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, blue_mask = cv2.threshold(blue_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Green Mask", green_mask)
    cv2.imshow("Blue Mask", blue_mask)
    cv2.waitKey(10000)
    return red_mask, green_mask, blue_mask

def find_and_locate_objects(image_to_process):
    red_mask, green_mask, blue_mask = hsv_color_segmentation(image_to_process)
    color_masks = {"red": red_mask, "green": green_mask, "blue": blue_mask}
    found_objects = []
    for color_name, mask in color_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500: continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            sides = len(approx)
            shape = ""
            if sides == 4: shape = "square"
            elif sides >= 7: shape = "circle"
            else: continue
            M = cv2.moments(contour)
            if M["m00"] == 0: continue
            cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
            try:
                relative_x, relative_y = vision.relative_pos_from_pixels(image_to_process, cX, cY)
                found_objects.append({
                    "color": color_name, "shape": shape, "pixel_center": (cX, cY),
                    "robot_position": (round(relative_x, 3), round(relative_y, 3))
                })
            except Exception as e:
                print(f"Error calculating robot position: {e}")
    return found_objects

def draw_object_info(image, objects_list):
    for obj in objects_list:
        cX, cY = obj["pixel_center"]
        label = f"{obj['color']} {obj['shape']}"
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, label, (cX - 40, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image

# --- Main Execution ---
robot = NiryoRobot("169.254.200.200")
robot.calibrate_auto()

# Define key positions
# picture_taking_position = PoseObject(0.12583522204120137, 0.0009559447500240502, 0.2690421753814516, 3.0810781986205438, 1.4793900090060592, 3.0801656604176415)
up_position = PoseObject(0.12583522204120137, 0.0009559447500240502, 0.3690421753814516, 3.0810781986205438, 1.4793900090060592, 3.0801656604176415)
picture_taking_position = PoseObject(0, -0.176, 0.27, 3, 1.33, 1.5)
place_position = PoseObject(x=0.0, y=-0.2, z=0.12, roll=0.0, pitch=1.57, yaw=1.57)

try:
    # Move to a neutral, observing position
    robot.move(picture_taking_position)
    robot.update_tool()

    # --- Start the Live Feed Loop ---
    # 1. CAPTURE AND PROCESS IMAGE
    camera_intrinsics, distortion_coefficients = robot.get_camera_intrinsics()
    original_image = uncompress_image(robot.get_img_compressed())
    image = vision.undistort_image(original_image, camera_intrinsics, distortion_coefficients)
    workspace_image = vision.extract_img_workspace(image)

    
    # if workspace_image is None:
    #     print("Could not extract workspace from the camera feed.")
    #     sleep(1)
    #     continue
        
    # 2. FIND AND DRAW OBJECTS
    detected_objects = find_and_locate_objects(workspace_image)
    image_with_detections = draw_object_info(workspace_image.copy(), detected_objects)

    # 3. DISPLAY THE LIVE FEED
    # cv2.imshow("Niryo Live Feed - Press 'p' to pick, 'q' to quit", image_with_detections)
    # key = cv2.waitKey(1) & 0xFF

    # 4. HANDLE KEYBOARD INPUT
    # if key == ord('q'):
    #     print("Quit command received.")
    #     break

    # if key == ord('p'):
    if True:
        if detected_objects:
            print("\n'p' key pressed. Initiating pick and place...")
            
            target_object = detected_objects[0] # Target the first object
            print(f"Targeting the {target_object['color'].upper()} {target_object['shape'].upper()}...")

            pick_x = target_object['robot_position'][0]
            pick_y = target_object['robot_position'][1]
            
            # IMPORTANT: Adjust this Z value for the height of your object!
            PICK_HEIGHT_Z = 0.015 
            GRIPPER_PITCH = 1.57

            x, y = vision.relative_pos_from_pixels(workspace_image, target_object['pixel_center'][0], target_object['pixel_center'][1])
            print("x: ", x, "y: ", y)

            pickup_pose = robot.get_target_pose_from_rel("table", 0.003, x, y, GRIPPER_PITCH)

            # Execute the pick and place sequence
            robot.release_with_tool()
            robot.move(pickup_pose)
            robot.grasp_with_tool()
            robot.move(up_position)
            robot.move(Beta_zone_drop_1)
            robot.release_with_tool()

            # Return to observing position to look for more objects
            robot.move(picture_taking_position)
            print("Pick and place complete. Looking for new objects...")
        else:
            print("\n'p' key pressed, but no objects were found to pick.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Safely clean up
    print("Exiting program.")
    cv2.destroyAllWindows()
    robot.move(picture_taking_position)
    robot.close_connection()