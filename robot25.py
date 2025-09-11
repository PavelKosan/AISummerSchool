from pyniryo import NiryoRobot, ObjectColor, ObjectShape
from pyniryo import vision
from pyniryo.vision.image_functions import relative_pos_from_pixels
import numpy as np

# Task 2.5

def detect_object_with_center_and_rotation():
	"""
	Function that:
	1. Fetches image workspace
	2. Performs object detection 
	3. Detects the center of the object in pixels and its rotation in radians
	4. Calculates the relative position of the object using PyNiryo's relative_pos_from_pixels method
	"""
	
	# Calibrate and get camera intrinsics
	robot.calibrate_auto()
	mtx, dist = robot.get_camera_intrinsics()
	
	# Step 1: Get image and extract workspace
	img_compressed = robot.get_img_compressed()
	img_raw = vision.uncompress_image(img_compressed)
	img_undistorted = vision.undistort_image(img_raw, mtx, dist)
	
	# Extract workspace from the undistorted image
	workspace_img = vision.extract_img_workspace(img_undistorted, 1)
	
	print("Step 1: Image workspace extracted successfully")
	
	# Step 2: Perform object detection using pyniryo methods
	# First try to detect using robot's built-in object detection
	obj_found, obj_rel_pos, obj_shape, obj_color = robot.detect_object(
		workspace_name=workspace_name, 
		color=ObjectColor.ANY,
		shape=ObjectShape.ANY
	)
	
	if obj_found:
		print(f"Step 2: Object detected - Color: {obj_color}, Shape: {obj_shape}")
		print(f"Object relative position: {obj_rel_pos}")
		
		# Step 3: Get center and rotation using vision functions
		# Based on the reference code provided
		
		# Try different colors to find the object
		colors_to_try = [vision.ColorHSV.BLUE, vision.ColorHSV.RED, vision.ColorHSV.GREEN, vision.ColorHSV.ANY]
		object_found = False
		center_x, center_y = 0, 0
		angle_degrees = 0
		
		for color_hsv in colors_to_try:
			try:
				# Threshold for the specific color using the reference pattern
				img_threshold = vision.threshold_hsv(workspace_img, *color_hsv.value)
				
				# Apply morphological operations using the reference pattern
				img_threshold = vision.morphological_transformations(img_threshold,
																   morpho_type=vision.MorphoType.OPEN,
																   kernel_shape=(11, 11),
																   kernel_type=vision.KernelType.ELLIPSE)
				
				# Find the biggest contour using the reference pattern
				cnt = vision.biggest_contour_finder(img_threshold)
				
				if cnt is not None and len(cnt) > 0:
					# Get center and angle using the reference pattern
					cnt_barycenter = vision.get_contour_barycenter(cnt)
					center_x, center_y = cnt_barycenter
					angle_degrees = vision.get_contour_angle(cnt)
					angle_radians = np.radians(angle_degrees)
					
					print(f"Step 3: Object center detected at pixels: ({center_x}, {center_y})")
					print(f"Object rotation: {angle_degrees} degrees ({angle_radians:.4f} radians)")
					
					object_found = True
					break
			except Exception as e:
				print(f"Error processing color {color_hsv}: {e}")
				continue
		
		if not object_found:
			print("Step 3: Could not determine object center and rotation from image processing")
			
		# Step 4: Calculate relative position using PyNiryo's relative_pos_from_pixels
		relative_pos_calculated = None
		if object_found and workspace_img is not None:
			try:
				# Convert pixel coordinates to relative workspace position
				relative_pos_calculated = relative_pos_from_pixels(workspace_img, center_x, center_y)
				print(f"Step 4: Relative position calculated from pixels: {relative_pos_calculated}")
			except Exception as e:
				print(f"Step 4: Error calculating relative position from pixels: {e}")
				relative_pos_calculated = None
		
		return {
			'workspace_img': workspace_img,
			'object_detected': obj_found,
			'object_color': obj_color,
			'object_shape': obj_shape,
			'relative_position': obj_rel_pos,
			'relative_position_from_pixels': relative_pos_calculated,
			'center_pixels': (center_x, center_y) if object_found else None,
			'rotation_radians': angle_radians if object_found else None,
			'rotation_degrees': angle_degrees if object_found else None
		}
		# return workspace_img, obj_found, obj_rel_pos, obj_shape, obj_color, relative_pos_calculated, (center_x, center_y), angle_radians, angle_degrees
	else:
		print("Step 2: No object detected in workspace")
		return {
			'workspace_img': None,
			'object_detected': False,
			'object_color': None,
			'object_shape': None,
			'relative_position': None,
			'relative_position_from_pixels': None,
			'center_pixels': None,
			'rotation_radians': None,
			'rotation_degrees': None
		}


if _name_ == '_main_':

	# show_image_stream_from_robot()

	# found , rel_pos, shape, colour =  robot.detect_object(workspace_name=workspace_name)
	# print(f"Object found: {found}")
	# print(f"Object relative position: {rel_pos}")
	# print(f"Object shape: {shape}")
	# print(f"Object colour: {colour}")
	result = detect_object_with_center_and_rotation()
	print("\n=== Detection Results ===")
	print(f"Object detected: {result['object_detected']}")
	if result['object_detected']:
		print(f"Object color: {result['object_color']}")
		print(f"Object shape: {result['object_shape']}")
		print(f"Relative position (robot method): {result['relative_position']}")
		print(f"Relative position (from pixels): {result['relative_position_from_pixels']}")
		print(f"Center (pixels): {result['center_pixels']}")
		print(f"Rotation (radians): {result['rotation_radians']}")
		print(f"Rotation (degrees): {result['rotation_degrees']}")
	
	# Uncomment to test other functions
	#show_image_stream_from_robot()
	#move_to_object()
	robot.close_connection()