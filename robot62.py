from pyniryo import *
from llm_utils import llmollama
import cv2
import time

# LLM Configuration
OLLAMA_HOST = "https://ollama.kky.zcu.cz"
OLLAMA_USER = "niryo"
OLLAMA_PASS = "Aec3aiqu3oodahye"

# Initialize robot
robot = NiryoRobot('169.254.200.200')

# Define placement areas (adjust coordinates based on your workspace)
PLACEMENT_AREAS = {
    1: {"x": 0.20, "y": -0.10, "z": 0.12},  # Area 1
    2: {"x": 0.20, "y": 0.0, "z": 0.12},    # Area 2
    3: {"x": 0.20, "y": 0.10, "z": 0.12},   # Area 3
}

def capture_workspace_image():
    """Capture and save an image of the workspace"""
    # Move to observation position
    obs_joints = robot.get_joints()
    obs_joints[0] = obs_joints[0] - 1.6  # Adjust x
    obs_joints[2] = obs_joints[2] + 1    # Adjust z
    obs_joints[4] = obs_joints[4] - 1.7  # Adjust rotation
    robot.move(obs_joints)

    # Capture and process image
    mtx, dist = robot.get_camera_intrinsics()
    img_compressed = robot.get_img_compressed()
    img = vision.uncompress_image(img_compressed)
    img = vision.undistort_image(img, mtx, dist)
    workspace_img = vision.extract_img_workspace(img, 1)

    # Save the image
    image_path = "workspace_current.jpg"
    cv2.imwrite(image_path, workspace_img)
    return image_path

def get_object_info_from_llm(image_path, text):
    """Get object information using LLM.
    Returns list of (color, shape, area_number) tuples
    """
    try:
        result = llmollama(image_path, OLLAMA_HOST, OLLAMA_USER, OLLAMA_PASS, user_prompt=text)
        
        if isinstance(result, list):
            objects = []
            for idx, obj in enumerate(result):
                # Assign area numbers cyclically to objects (1, 2, 3, 1, 2, 3, ...)
                area_number = (idx % len(PLACEMENT_AREAS)) + 1
                objects.append((obj.get("color"), obj.get("shape"), area_number))
            return objects
        else:
            print("[WARN] Unexpected result type:", type(result))
            return None

    except Exception as e:
        print(f"Error getting object info from LLM: {e}")
        return None

def pick_object(color, shape):
    """Pick up the identified object"""
    try:
        # Convert string color to ObjectColor enum
        color_map = {
            "red": ObjectColor.RED,
            "blue": ObjectColor.BLUE,
            "green": ObjectColor.GREEN
        }
        robot_color = color_map.get(color.lower(), ObjectColor.ANY)

        # Convert string shape to ObjectShape enum
        shape_map = {
            "square": ObjectShape.SQUARE,
            "circle": ObjectShape.CIRCLE
        }
        robot_shape = shape_map.get(shape.lower(), ObjectShape.ANY)

        # Attempt to pick up the object
        obj_found, shape_ret, color_ret = robot.vision_pick(
            workspace_name="Tobor",
            height_offset=0.0,
            shape=robot_shape,
            color=robot_color
        )

        return obj_found
    except Exception as e:
        print(f"Error during pick operation: {e}")
        return False

def place_in_area(area_number):
    """Place the object in the specified numbered area"""
    try:
        if area_number not in PLACEMENT_AREAS:
            print(f"Invalid area number: {area_number}")
            return False

        area = PLACEMENT_AREAS[area_number]
        
        # Move to the specified area
        robot.move_pose([area["x"], area["y"], area["z"], 0.0, 1.57, 0.0])
        
        # Release the object
        robot.release_with_tool()
        
        # Move up slightly to avoid collision
        robot.move_pose([area["x"], area["y"], area["z"] + 0.05, 0.0, 1.57, 0.0])
        
        return True
    except Exception as e:
        print(f"Error during place operation: {e}")
        return False

def advanced_pick_and_place(prompt):
    """Main function to execute the advanced pick and place operation using LLM"""
    try:
        # Initial setup
        robot.calibrate_auto()
        robot.move_to_home_pose()
        robot.clear_collision_detected()

        # Capture workspace image
        print("Capturing workspace image...")
        image_path = capture_workspace_image()

        # Get object information from LLM
        print("Analyzing image with LLM...")
        objects = get_object_info_from_llm(image_path, prompt)
        
        if not objects:
            print("No objects detected by LLM")
            return False

        # Process each detected object
        for color, shape, area_number in objects:
            print(f"Processing: {color} {shape} -> Area {area_number}")

            # Pick up the object
            print(f"Attempting to pick up {color} {shape}...")
            if not pick_object(color, shape):
                print("Failed to pick up the object")
                continue

            # Place in designated area
            print(f"Placing object in area {area_number}...")
            if not place_in_area(area_number):
                print("Failed to place object in designated area")
                continue

            # Return to home position between objects
            robot.move_to_home_pose()
            print(f"Successfully placed {color} {shape} in area {area_number}")

        print("Advanced pick and place operation completed")
        return True

    except Exception as e:
        print(f"Error during advanced pick and place operation: {e}")
        return False

if __name__ == "__main__":
    try:
        robot.clear_collision_detected()
        # Example prompts:
        # "List all red objects in the workspace"
        # "Find all square objects regardless of color"
        # "Detect all objects and sort them by color"
        advanced_pick_and_place("List all objects in the workspace, sorted by color (red, then blue, then green)")
    finally:
        robot.move_to_home_pose()
        robot.close_connection()
