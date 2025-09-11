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
# Set up conveyor
conveyor_id = robot.set_conveyor()

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

def get_object_info_from_llm(image_path):
    """Get object information using LLM"""
    try:
        result = llmollama(image_path, OLLAMA_HOST, OLLAMA_USER, OLLAMA_PASS)
        return result.color, result.shape
    except Exception as e:
        print(f"Error getting object info from LLM: {e}")
        return None, None

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

def place_on_conveyor():
    """Place the object on the conveyor belt"""
    try:
        # Move to a position above the conveyor
        cur_joints = robot.get_joints()
        cur_joints[1] = cur_joints[1] - 0.3  # Adjust y for conveyor position
        robot.move(cur_joints)
        
        # Release the object
        robot.release_with_tool()
        
        # Move back slightly
        cur_joints[1] = cur_joints[1] + 0.3
        robot.move(cur_joints)
        
        # Start conveyor movement
        robot.run_conveyor(conveyor_id, 50, ConveyorDirection.FORWARD)
        time.sleep(2)  # Run for 2 seconds
        robot.stop_conveyor(conveyor_id)
        
        return True
    except Exception as e:
        print(f"Error during place operation: {e}")
        return False

def pick_and_place_with_llm():
    """Main function to execute the pick and place operation using LLM"""
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
        color, shape = get_object_info_from_llm(image_path)
        if not color or not shape:
            print("Could not identify object with LLM")
            return False
        
        print(f"LLM detected: {color} {shape}")
        
        # Pick up the object
        print("Attempting to pick up the object...")
        if not pick_object(color, shape):
            print("Failed to pick up the object")
            return False
        
        # Place on conveyor
        print("Placing object on conveyor...")
        if not place_on_conveyor():
            print("Failed to place object on conveyor")
            return False
        
        # Return to home position
        robot.move_to_home_pose()
        print("Pick and place operation completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during pick and place operation: {e}")
        return False
    finally:
        # Ensure we stop the conveyor
        robot.stop_conveyor(conveyor_id)

if __name__ == "__main__":
    try:
        pick_and_place_with_llm()
    finally:
        # Clean up
        robot.stop_conveyor(conveyor_id)
        robot.move_to_home_pose()
        robot.close_connection()
