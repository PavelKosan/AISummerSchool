from pyniryo import *
import time

robot = NiryoRobot('169.254.200.200')



def calibrate():
    robot.calibrate_auto()

def blink():
    robot.led_ring_flashing([15, 50, 255])

def print_position():
    print(robot.get_pose())

def print_position2():
    print(robot.get_joints())


def print_position_in_loop():
    robot.calibrate_auto()
    try:
        while True:
            print(robot.get_pose())
            time.sleep(1)
    except KeyboardInterrupt:
        print("keyboard interrupt... closing connection")
        robot.close_connection()

def first_move():
    robot.calibrate_auto()
    print("waiting 60 seconds")
    robot.wait(60)
    print("returning to home pose")
    robot.move_to_home_pose()

def move_between():
    robot.calibrate_auto()
    print("waiting 10 seconds for first position")
    robot.wait(10)
    firstPose = robot.get_pose()

    print("waiting 10 seconds for second position")
    robot.wait(10)
    secondPose = robot.get_pose()

    robot.set_arm_max_velocity(40)

    for i in range(1, 10):
        robot.move(firstPose)
        robot.wait(0.5)
        robot.move(secondPose)

def make_square():
    pos = robot.get_joints()
    print(pos)
    x = 0.2
    for i in range(1, 4):
        pos[0] = pos[0] - x
        robot.move(pos)

        pos[1] = pos[1] - x
        robot.move(pos)

        pos[0] = pos[0] + x
        robot.move(pos)

        pos[1] = pos[1] + x
        robot.move(pos)

        x -= 0.05

def grasp_in_square():
    pos = robot.get_joints()
    x = 0.2

    pos[0] = pos[0] - x
    robot.move(pos)
    robot.grasp_with_tool()

    pos[1] = pos[1] - x
    robot.move(pos)
    robot.release_with_tool()


    pos[0] = pos[0] + x
    robot.move(pos)
    robot.grasp_with_tool()

    pos[1] = pos[1] + x
    robot.move(pos)
    robot.release_with_tool()




#calibrate()
#blink()
#print_position()
#print_position2()
#print_position_in_loop()
#first_move()
#move_between()
#calibrate()
#make_square()
print(robot.get_pose())
#grasp_in_square()
robot.close_connection()