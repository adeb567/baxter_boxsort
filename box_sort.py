import rospy
import numpy as np
import cv2
import baxter_interface

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from cv_bridge import CvBridge

import struct

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

obj_color = None
obj_found = False
correct_location = True

lower_ranges = dict()
upper_ranges = dict() 
lower_ranges["red"] = np.array([0, 105, 51])
upper_ranges["red"] = np.array([179, 255, 255])
lower_ranges["green"] = np.array([42, 83, 0])
upper_ranges["green"] = np.array([86, 166, 76])
lower_ranges["blue"] = np.array([98, 99, 14])
upper_ranges["blue"] = np.array([121, 253, 43])


camera_matrix = np.array([[405.916530305, 0, 657.501404458],
                          [0, 405.916530305, 411.696520403],
                          [0, 0, 1]])

table_height = -0.24254757440653632

#Object centroid position in the baxter's stationary base frame 
xb = 0
yb = 0

count_red = 0
count_blue = 0
count_green = 0

do_cv = 0

def callback(message):
    global xb, yb, obj_color, obj_found
    
    if do_cv == 1:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(message, "bgr8")
        
        for color in lower_ranges.keys():
            frame = cv_image
            frame = cv2.resize(frame, (640, 400))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_ranges[color], upper_ranges[color])
            _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            _, cnts, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                x = 400
                if cv2.contourArea(c) > x:
                    x, y, w, h = cv2.boundingRect(c)
                    cx = x+w*0.5
                    cy = y+h*0.5

                    cx = 320 - cx
                    cy = 200 - cy
  
                    obj_color = color

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, ("DETECT"), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    global xb, yb
                    xb= 0.5813876751804535 + cy*0.00108 - 0.01
                    yb= -0.1833511949521925 + cx*0.0011 + 0.025

            cv2.imshow("FRAME", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break



def main():
    global xb, yb, do_cv
    #Initiate  node
    rospy.init_node('pnp')


    #Subscribe to right hand camera image 
    rospy.Subscriber("/cameras/right_hand_camera/image", Image, callback)
    rospy.sleep(3)
    
    right_arm = baxter_interface.Limb('right')
    gc_r = baxter_interface.Gripper('right')
    gc_r.calibrate()

    while True:
        right_joint_angles = ik_test('right', 'home')
        right_arm.move_to_joint_positions(right_joint_angles)
        gc_r.open()
        
        do_cv = 1
        rospy.sleep(3)
        do_cv = 0

        right_joint_angles = ik_test('right', 'approach')
        right_arm.move_to_joint_positions(right_joint_angles)

        right_joint_angles = ik_test('right', 'pick')
        right_arm.move_to_joint_positions(right_joint_angles)
        gc_r.close()

        right_joint_angles = ik_test('right', 'home')
        right_arm.move_to_joint_positions(right_joint_angles)

        if obj_color == "red":
            right_joint_angles = ik_test('right', 'red')
        elif obj_color == "blue":
            right_joint_angles = ik_test('right', 'blue')
        elif obj_color == "green":
            right_joint_angles = ik_test('right', 'green')
        elif obj_color == "None":
            break

        right_arm.move_to_joint_positions(right_joint_angles)
        gc_r.open()
        
        right_joint_angles = ik_test('right', 'drop')
        right_arm.move_to_joint_positions(right_joint_angles)



def ik_test(limb, case):
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        'right': PoseStamped(
            header=hdr,
            pose=get_pose(case),
            ),
    }

    ikreq.pose_stamp.append(poses[limb])
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1

    # Check if result valid, and type of seed ultimately used to get solution
    # convert rospy's string representation of uint8[]'s to int's
    resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                               resp.result_type)
    if (resp_seeds[0] != resp.RESULT_INVALID):
        seed_str = {
                    ikreq.SEED_USER: 'User Provided Seed',
                    ikreq.SEED_CURRENT: 'Current Joint Angles',
                    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                   }.get(resp_seeds[0], 'None')
        print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
              (seed_str,))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        print("\nIK Joint Solution:\n", limb_joints)
        return limb_joints
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")
        return {}
    


def get_pose(case):
    global xb,yb
    if case == 'home':
        return Pose(
                position=Point(
                    x=0.5813876751804535,
                    y=-0.1833511949521925,
                    z=0.09941425665002981,
                ),
                orientation=Quaternion(
                    x=0,
                    y=1,
                    z=0,
                    w=0,
                )
        )
    if case == 'red':
        global count_red
        count_red += 0.03
        return Pose(
                position=Point(
                    x=0.6177436108949574,
                    y=-0.6325383787467281,
                    z= table_height + count_red,
                ),
                orientation=Quaternion(
                    x=0,
                    y=1,
                    z=0,
                    w=0,
                )
        )
    if case == 'blue':
        global count_blue
        count_blue += 0.03
        return Pose(
                position=Point(
                    x=0.5177436108949574,
                    y=-0.6325383787467281,
                    z= table_height + count_blue,
                ),
                orientation=Quaternion(
                    x=0,
                    y=1,
                    z=0,
                    w=0,
                )
        )
    if case == 'green':
        global count_green
        count_green += 0.03
        return Pose(
                position=Point(
                    x=0.7177436108949574,
                    y=-0.6325383787467281,
                    z= table_height + count_green,
                ),
                orientation=Quaternion(
                    x=0,
                    y=1,
                    z=0,
                    w=0,
                )
        )
    if case == 'pick':
        print(xb, yb)
        p = Pose(
                position=Point(
                    x= xb,
                    y= yb,
                    z= table_height,
                ),
                orientation=Quaternion(
                    x=0,
                    y=1,
                    z=0,
                    w=0,
                )
        )
        return p

    if case == 'approach':
        return Pose(
                position=Point(
                    x= xb,
                    y= yb,
                    z=-0.07054757440653632,
                ),
                orientation=Quaternion(
                    x=0,
                    y=1,
                    z=0,
                    w=0,
                )
            )
    


if __name__ == '__main__':
     main()


