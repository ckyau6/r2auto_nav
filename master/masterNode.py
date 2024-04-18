WHITE_FLAG = False

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8, UInt16, Float64, String, Int8
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import math
import heapq

import argparse
from collections import deque
from skimage.morphology import dilation, disk
from functools import cmp_to_key

from scipy.ndimage import generic_filter
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import dijkstra

import time

import cv2

# used to convert the occupancy grid to an image of map, umpapped, occupied
import scipy.stats

occ_bins = [-1, 0, 50, 100]

# CLEARANCE_RADIUS is in cm, used to dilate the obstacles
# radius of turtle bot is around 11 cm
# CLEARANCE_RADIUS = 5
CLEARANCE_RADIUS = 0

# this is in pixel
# FRONTIER_THRESHOLD = 4
FRONTIER_THRESHOLD = 3
# PIXEL_DEST_THRES = 2
PIXEL_DEST_THRES = 3

NAV_TOO_CLOSE = 0.23
BUTT_TOO_CLOSE = 0.10

BUCKET_TOO_CLOSE = 0.30

BUCKET_FRONT_RANGE = 10
BUCKET_FRONT_LEFT_ANGLE = 0 + BUCKET_FRONT_RANGE
BUCKET_FRONT_RIGHT_ANGLE = 360 - BUCKET_FRONT_RANGE

LEFT_RIGHT_ANGLE_RANGE = 5
LEFT_UPPER_ANGLE = 90 - LEFT_RIGHT_ANGLE_RANGE
LEFT_LOWER_ANGLE = 90 + LEFT_RIGHT_ANGLE_RANGE
RIGHT_LOWER_ANGLE = 270 - LEFT_RIGHT_ANGLE_RANGE
RIGHT_UPPER_ANGLE = 270 + LEFT_RIGHT_ANGLE_RANGE

BACK_ANGLE_RANGE = 5
BACK_LOWER_ANGLE = 180 - BACK_ANGLE_RANGE
BACK_UPPER_ANGLE = 180 + BACK_ANGLE_RANGE

MAZE_FRONT_RANGE = 26
MAZE_FRONT_LEFT_ANGLE = 0 + MAZE_FRONT_RANGE
MAZE_FRONT_RIGHT_ANGLE = 360 - MAZE_FRONT_RANGE

MAZE_CLEARANCE_ANGLE = 10
MAZE_ROTATE_SPEED = 64

UNMAPPED = 1
OPEN = 2
OBSTACLE = 3

# this is for path finder to ignore close points, in pixels
# RADIUS_OF_IGNORE = 3
RADIUS_OF_IGNORE = 1

PARAMETER_R = 0.93
# use odd number for window size
# WINDOWSIZE = 21
# WINDOWSIZE = 9
WINDOWSIZE = 25

# if distance is more than this value, skip that point
# FRONTIER_SKIP_THRESHOLD = 1e9
FRONTIER_SKIP_THRESHOLD = 3e9

# distance to two points to be considered sorted by lower y, in meter
FRONTIER_DIST_M = 0.10

# radius around bot to be visited, in meter
VISIT_RADIUS_M = 0.5

# time in s to wait after no more frontier before goin to hall way
WAIT_FRONTIER = 5

# cost to avoid must visit points that is across the wall
VISIT_COST = 1e9

# left, right door and finish line coords in meters from the magic origin (starting point roughly 20cm from walls three sides)
LEFT_DOOR_COORDS_M = (1.20-0.35, 2.70)
RIGHT_DOOR_COORDS_M = (1.90+0.35, 2.70)
FINISH_LINE_M = ((LEFT_DOOR_COORDS_M[0] + RIGHT_DOOR_COORDS_M[0])/2, 2.10)

# must check points
# as doing frontier searching, if one of these points are better than other frontier, frontier will give these points
# there will be cost check on these points incase it landed on a unreachable point due to maze elements, but the cost to give up must be much higher than frontier (cannot give these points up unless really necessary)
# these points are cleared if the pixel is mapped (OPEN or OBSTACLE)

# map is X = 3.5, Y = 3.5, with room at y = 2.1

# make a dot grid that spans X = 3.5 and Y = 2.1, starting from 0.2, 0.2
# Define the start, end, and step for X and Y
start_x, end_x, no_x = 0, 3.1, 30
start_y, end_y, no_y = 0, 2.05, 30

# Create the grid, must be last item bigges x and y
MUST_VISIT_POINTS_M = [(x, y) for y in np.linspace(start_y, end_y, no_y) for x in np.linspace(start_x, end_x, no_x)]

# all
# MUST_VISIT_POINTS_M = [(0.00, 1.70), (0.34, 1.70), (0.69, 1.70), (1.03, 1.70), (1.38, 1.70), (1.72, 1.70), (2.07, 1.70), (2.41, 1.70), (2.76, 1.70), (3.10, 1.70), (2.7, 1.6),
#                        (0.00, 1.51), (0.34, 1.51), (0.69, 1.51), (1.03, 1.51), (1.38, 1.51), (1.72, 1.51), (2.07, 1.51), (2.41, 1.51), (2.76, 1.51), (3.10, 1.51), 
#                        (0.00, 1.32), (0.34, 1.32), (0.69, 1.32), (1.03, 1.32), (1.38, 1.32), (1.72, 1.32), (2.07, 1.32), (2.41, 1.32), (2.76, 1.32), (3.10, 1.32), 
#                        (0.00, 1.13), (0.34, 1.13), (0.69, 1.13), (1.03, 1.13), (1.38, 1.13), (1.72, 1.13), (2.07, 1.13), (2.41, 1.13), (2.76, 1.13), (3.10, 1.13), 
#                        (0.00, 0.94), (0.34, 0.94), (0.69, 0.94), (1.03, 0.94), (1.38, 0.94), (1.72, 0.94), (2.07, 0.94), (2.41, 0.94), (2.76, 0.94), (3.10, 0.94), 
#                        (0.00, 0.76), (0.34, 0.76), (0.69, 0.76), (1.03, 0.76), (1.38, 0.76), (1.72, 0.76), (2.07, 0.76), (2.41, 0.76), (2.76, 0.76), (3.10, 0.76), 
#                        (0.00, 0.57), (0.34, 0.57), (0.69, 0.57), (1.03, 0.57), (1.38, 0.57), (1.72, 0.57), (2.07, 0.57), (2.41, 0.57), (2.76, 0.57), (3.10, 0.57), 
#                        (0.00, 0.38), (0.34, 0.38), (0.69, 0.38), (1.03, 0.38), (1.38, 0.38), (1.72, 0.38), (2.07, 0.38), (2.41, 0.38), (2.76, 0.38), (3.10, 0.38), 
#                        (0.00, 0.19), (0.34, 0.19), (0.69, 0.19), (1.03, 0.19), (1.38, 0.19), (1.72, 0.19), (2.07, 0.19), (2.41, 0.19), (2.76, 0.19), (3.10, 0.19), 
#                        (0.00, 0.00), (0.34, 0.00), (0.69, 0.00), (1.03, 0.00), (1.38, 0.00), (1.72, 0.00), (2.07, 0.00), (2.41, 0.00), (2.76, 0.00), (3.10, 0.00)]

# remove top left
# MUST_VISIT_POINTS_M = [                                          (1.03, 1.70), (1.38, 1.70), (1.72, 1.70), (2.07, 1.70), (2.41, 1.70), (2.76, 1.70), (3.10, 1.70),
#                                                                  (1.03, 1.51), (1.38, 1.51), (1.72, 1.51), (2.07, 1.51), (2.41, 1.51), (2.76, 1.51), (3.10, 1.51), 
#                                                                  (1.03, 1.32), (1.38, 1.32), (1.72, 1.32), (2.07, 1.32), (2.41, 1.32), (2.76, 1.32), (3.10, 1.32), 
#                        (0.00, 1.13), (0.34, 1.13), (0.69, 1.13), (1.03, 1.13), (1.38, 1.13), (1.72, 1.13), (2.07, 1.13), (2.41, 1.13), (2.76, 1.13), (3.10, 1.13), 
#                        (0.00, 0.94), (0.34, 0.94), (0.69, 0.94), (1.03, 0.94), (1.38, 0.94), (1.72, 0.94), (2.07, 0.94), (2.41, 0.94), (2.76, 0.94), (3.10, 0.94), 
#                        (0.00, 0.76), (0.34, 0.76), (0.69, 0.76), (1.03, 0.76), (1.38, 0.76), (1.72, 0.76), (2.07, 0.76), (2.41, 0.76), (2.76, 0.76), (3.10, 0.76), 
#                        (0.00, 0.57), (0.34, 0.57), (0.69, 0.57), (1.03, 0.57), (1.38, 0.57), (1.72, 0.57), (2.07, 0.57), (2.41, 0.57), (2.76, 0.57), (3.10, 0.57), 
#                        (0.00, 0.38), (0.34, 0.38), (0.69, 0.38), (1.03, 0.38), (1.38, 0.38), (1.72, 0.38), (2.07, 0.38), (2.41, 0.38), (2.76, 0.38), (3.10, 0.38), 
#                        (0.00, 0.19), (0.34, 0.19), (0.69, 0.19), (1.03, 0.19), (1.38, 0.19), (1.72, 0.19), (2.07, 0.19), (2.41, 0.19), (2.76, 0.19), (3.10, 0.19), 
#                        (0.00, 0.00), (0.34, 0.00), (0.69, 0.00), (1.03, 0.00), (1.38, 0.00), (1.72, 0.00), (2.07, 0.00), (2.41, 0.00), (2.76, 0.00), (3.10, 0.00)]


# MUST_VISIT_POINTS_M = \
#     [(0.00, 1.70), (0.16, 1.70), (0.33, 1.70), (0.49, 1.70), (0.65, 1.70), (0.82, 1.70), (0.98, 1.70), (1.14, 1.70), (1.31, 1.70), (1.47, 1.70), (1.63, 1.70), (1.79, 1.70), (1.96, 1.70), (2.12, 1.70), (2.28, 1.70), (2.45, 1.70), (2.61, 1.70), (2.77, 1.70), (2.94, 1.70), (3.1, 1.70),
#      (0.00, 1.61), (0.16, 1.61), (0.33, 1.61), (0.49, 1.61), (0.65, 1.61), (0.82, 1.61), (0.98, 1.61), (1.14, 1.61), (1.31, 1.61), (1.47, 1.61), (1.63, 1.61), (1.79, 1.61), (1.96, 1.61), (2.12, 1.61), (2.28, 1.61), (2.45, 1.61), (2.61, 1.61), (2.77, 1.61), (2.94, 1.61), (3.1, 1.61), 
#      (0.00, 1.52), (0.16, 1.52), (0.33, 1.52), (0.49, 1.52), (0.65, 1.52), (0.82, 1.52), (0.98, 1.52), (1.14, 1.52), (1.31, 1.52), (1.47, 1.52), (1.63, 1.52), (1.79, 1.52), (1.96, 1.52), (2.12, 1.52), (2.28, 1.52), (2.45, 1.52), (2.61, 1.52), (2.77, 1.52), (2.94, 1.52), (3.1, 1.52), 
#      (0.00, 1.43), (0.16, 1.43), (0.33, 1.43), (0.49, 1.43), (0.65, 1.43), (0.82, 1.43), (0.98, 1.43), (1.14, 1.43), (1.31, 1.43), (1.47, 1.43), (1.63, 1.43), (1.79, 1.43), (1.96, 1.43), (2.12, 1.43), (2.28, 1.43), (2.45, 1.43), (2.61, 1.43), (2.77, 1.43), (2.94, 1.43), (3.1, 1.43), 
#      (0.00, 1.34), (0.16, 1.34), (0.33, 1.34), (0.49, 1.34), (0.65, 1.34), (0.82, 1.34), (0.98, 1.34), (1.14, 1.34), (1.31, 1.34), (1.47, 1.34), (1.63, 1.34), (1.79, 1.34), (1.96, 1.34), (2.12, 1.34), (2.28, 1.34), (2.45, 1.34), (2.61, 1.34), (2.77, 1.34), (2.94, 1.34), (3.1, 1.34), 
#      (0.00, 1.25), (0.16, 1.25), (0.33, 1.25), (0.49, 1.25), (0.65, 1.25), (0.82, 1.25), (0.98, 1.25), (1.14, 1.25), (1.31, 1.25), (1.47, 1.25), (1.63, 1.25), (1.79, 1.25), (1.96, 1.25), (2.12, 1.25), (2.28, 1.25), (2.45, 1.25), (2.61, 1.25), (2.77, 1.25), (2.94, 1.25), (3.1, 1.25), 
#      (0.00, 1.16), (0.16, 1.16), (0.33, 1.16), (0.49, 1.16), (0.65, 1.16), (0.82, 1.16), (0.98, 1.16), (1.14, 1.16), (1.31, 1.16), (1.47, 1.16), (1.63, 1.16), (1.79, 1.16), (1.96, 1.16), (2.12, 1.16), (2.28, 1.16), (2.45, 1.16), (2.61, 1.16), (2.77, 1.16), (2.94, 1.16), (3.1, 1.16), 
#      (0.00, 1.07), (0.16, 1.07), (0.33, 1.07), (0.49, 1.07), (0.65, 1.07), (0.82, 1.07), (0.98, 1.07), (1.14, 1.07), (1.31, 1.07), (1.47, 1.07), (1.63, 1.07), (1.79, 1.07), (1.96, 1.07), (2.12, 1.07), (2.28, 1.07), (2.45, 1.07), (2.61, 1.07), (2.77, 1.07), (2.94, 1.07), (3.1, 1.07), 
#      (0.00, 0.98), (0.16, 0.98), (0.33, 0.98), (0.49, 0.98), (0.65, 0.98), (0.82, 0.98), (0.98, 0.98), (1.14, 0.98), (1.31, 0.98), (1.47, 0.98), (1.63, 0.98), (1.79, 0.98), (1.96, 0.98), (2.12, 0.98), (2.28, 0.98), (2.45, 0.98), (2.61, 0.98), (2.77, 0.98), (2.94, 0.98), (3.1, 0.98), 
#      (0.00, 0.89), (0.16, 0.89), (0.33, 0.89), (0.49, 0.89), (0.65, 0.89), (0.82, 0.89), (0.98, 0.89), (1.14, 0.89), (1.31, 0.89), (1.47, 0.89), (1.63, 0.89), (1.79, 0.89), (1.96, 0.89), (2.12, 0.89), (2.28, 0.89), (2.45, 0.89), (2.61, 0.89), (2.77, 0.89), (2.94, 0.89), (3.1, 0.89), 
#      (0.00, 0.81), (0.16, 0.81), (0.33, 0.81), (0.49, 0.81), (0.65, 0.81), (0.82, 0.81), (0.98, 0.81), (1.14, 0.81), (1.31, 0.81), (1.47, 0.81), (1.63, 0.81), (1.79, 0.81), (1.96, 0.81), (2.12, 0.81), (2.28, 0.81), (2.45, 0.81), (2.61, 0.81), (2.77, 0.81), (2.94, 0.81), (3.1, 0.81), 
#      (0.00, 0.72), (0.16, 0.72), (0.33, 0.72), (0.49, 0.72), (0.65, 0.72), (0.82, 0.72), (0.98, 0.72), (1.14, 0.72), (1.31, 0.72), (1.47, 0.72), (1.63, 0.72), (1.79, 0.72), (1.96, 0.72), (2.12, 0.72), (2.28, 0.72), (2.45, 0.72), (2.61, 0.72), (2.77, 0.72), (2.94, 0.72), (3.1, 0.72), 
#      (0.00, 0.63), (0.16, 0.63), (0.33, 0.63), (0.49, 0.63), (0.65, 0.63), (0.82, 0.63), (0.98, 0.63), (1.14, 0.63), (1.31, 0.63), (1.47, 0.63), (1.63, 0.63), (1.79, 0.63), (1.96, 0.63), (2.12, 0.63), (2.28, 0.63), (2.45, 0.63), (2.61, 0.63), (2.77, 0.63), (2.94, 0.63), (3.1, 0.63), 
#      (0.00, 0.54), (0.16, 0.54), (0.33, 0.54), (0.49, 0.54), (0.65, 0.54), (0.82, 0.54), (0.98, 0.54), (1.14, 0.54), (1.31, 0.54), (1.47, 0.54), (1.63, 0.54), (1.79, 0.54), (1.96, 0.54), (2.12, 0.54), (2.28, 0.54), (2.45, 0.54), (2.61, 0.54), (2.77, 0.54), (2.94, 0.54), (3.1, 0.54), 
#      (0.00, 0.45), (0.16, 0.45), (0.33, 0.45), (0.49, 0.45), (0.65, 0.45), (0.82, 0.45), (0.98, 0.45), (1.14, 0.45), (1.31, 0.45), (1.47, 0.45), (1.63, 0.45), (1.79, 0.45), (1.96, 0.45), (2.12, 0.45), (2.28, 0.45), (2.45, 0.45), (2.61, 0.45), (2.77, 0.45), (2.94, 0.45), (3.1, 0.45), 
#      (0.00, 0.36), (0.16, 0.36), (0.33, 0.36), (0.49, 0.36), (0.65, 0.36), (0.82, 0.36), (0.98, 0.36), (1.14, 0.36), (1.31, 0.36), (1.47, 0.36), (1.63, 0.36), (1.79, 0.36), (1.96, 0.36), (2.12, 0.36), (2.28, 0.36), (2.45, 0.36), (2.61, 0.36), (2.77, 0.36), (2.94, 0.36), (3.1, 0.36), 
#      (0.00, 0.27), (0.16, 0.27), (0.33, 0.27), (0.49, 0.27), (0.65, 0.27), (0.82, 0.27), (0.98, 0.27), (1.14, 0.27), (1.31, 0.27), (1.47, 0.27), (1.63, 0.27), (1.79, 0.27), (1.96, 0.27), (2.12, 0.27), (2.28, 0.27), (2.45, 0.27), (2.61, 0.27), (2.77, 0.27), (2.94, 0.27), (3.1, 0.27), 
#      (0.00, 0.18), (0.16, 0.18), (0.33, 0.18), (0.49, 0.18), (0.65, 0.18), (0.82, 0.18), (0.98, 0.18), (1.14, 0.18), (1.31, 0.18), (1.47, 0.18), (1.63, 0.18), (1.79, 0.18), (1.96, 0.18), (2.12, 0.18), (2.28, 0.18), (2.45, 0.18), (2.61, 0.18), (2.77, 0.18), (2.94, 0.18), (3.1, 0.18), 
#      (0.00, 0.09), (0.16, 0.09), (0.33, 0.09), (0.49, 0.09), (0.65, 0.09), (0.82, 0.09), (0.98, 0.09), (1.14, 0.09), (1.31, 0.09), (1.47, 0.09), (1.63, 0.09), (1.79, 0.09), (1.96, 0.09), (2.12, 0.09), (2.28, 0.09), (2.45, 0.09), (2.61, 0.09), (2.77, 0.09), (2.94, 0.09), (3.1, 0.09), 
#      (0.00, 0.00), (0.16, 0.00), (0.33, 0.00), (0.49, 0.00), (0.65, 0.00), (0.82, 0.00), (0.98, 0.00), (1.14, 0.00), (1.31, 0.00), (1.47, 0.00), (1.63, 0.00), (1.79, 0.00), (1.96, 0.00), (2.12, 0.00), (2.28, 0.00), (2.45, 0.00), (2.61, 0.00), (2.77, 0.00), (2.94, 0.00), (3.1, 0.00)]

MUST_VISIT_COST = FRONTIER_SKIP_THRESHOLD * 10000

# speedssss
LIN_MAX = 110
LIN_WHEN_ROTATING = 50

# return the rotation angle around z axis in degrees (counterclockwise)
def angle_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(t3, t4))


class MasterNode(Node):
    def __init__(self, show_plot):
        super().__init__('masterNode')
        self.show_plot = (show_plot == 'y')

        ''' ================================================ http request ================================================ '''
        # Create a subscriber to the topic "doorStatus"
        # Listens for the doorStatus from the doorRequestNode
        self.http_subscription = self.create_subscription(
            String,
            'doorStatus',
            self.http_listener_callback,
            10)
        self.http_subscription  # prevent unused variable warning

        # variable to be used to store the doorStatus
        self.doorStatus = ""

        # Create a publisher to the topic "doorRequest"
        # Publishes the door opening request to the doorRequestNode
        self.http_publisher = self.create_publisher(String, 'doorRequest', 10)

        ''' ================================================ limit switch ================================================ '''
        # Create a subscriber to the topic "switchStatus"
        # Listens for the switchStatus from the limitSwitchNode
        self.switch_subscription = self.create_subscription(
            String,
            'switchStatus',
            self.switch_listener_callback,
            10)
        self.switch_subscription  # prevent unused variable warning

        # variable to be used to store the limit switch status
        self.switchStatus = ""

        # Create a publisher to the topic "switchRequest"
        # Publishes the activate/deacivate request to the limitSwitchNode
        self.switch_publisher = self.create_publisher(String, 'switchRequest', 10)

        ''' ================================================ servo control ================================================ '''
        # Create a publisher to the topic "servoRequest"
        # Publishes the servoRequest to the servoControlNode
        self.servo_publisher = self.create_publisher(UInt8, 'servoRequest', 10)

        ''' ================================================ lidar ================================================ '''
        # Create a subscriber to the topic "scan"
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data)
        self.scan_subscription  # prevent unused variable warning
        self.laser_range = np.array([])
        
        self.bucketFrontLeftIndex = 0
        self.bucketFrontRightIndex = 0

        self.leftIndexL = 0
        self.leftIndexH = 0

        self.rightIndexL = 0
        self.rightIndexH = 0

        self.backIndexL = 0
        self.backIndexH = 0

        self.mazeFrontLeftindex = 0
        self.mazeFrontRightindex = 0

        ''' ================================================ bucket ================================================ '''
        # Listens for the bucket angle
        self.bucketAngle_subscription = self.create_subscription(
            UInt16,
            'bucketAngle',
            self.bucketAngle_listener_callback,
            10)
        self.bucketAngle_subscription  # prevent unused variable warning

        self.bucketAngle = 0
        
        self.bucketArray = [0, 0, 0, 0, 0]

        ''' ================================================ occupancy map ================================================ '''
        # Create a subscriber to the topic "map"
        self.occ_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occ_callback,
            qos_profile_sensor_data)
        self.occ_subscription  # prevent unused variable warning
        self.occupancyMap = self.dilutedOccupancyMap = self.frontierMap = np.array([])

        self.yaw = 0
        self.map_res = 0.05
        self.map_w = self.map_h = 0

        self.leftDoor_pixel = (0, 0)
        self.rightDoor_pixel = (0, 0)
        self.finishLine_Ypixel = 0
        
        self.disableOCC = False
        
        self.newOccFlag = False
        
        # the only ways for occ to be enabled and newOccFlag to be set to False are:
        #   too close to wall
        #   path is finished (this is only in maze_moving. move_to_hallway, go_to_left_door, go_to_right_door uses maze_moving)
        
        
        # occ is disabled after one new map is received and newOccFlag set to true
        
        # if occ is enabled but newOccFlag is false, then it will wait until newOccFlag is true
        # this is needed for all states that uses occ map, so the check is right before FSM
        
        # after entering room, occ is disabled and will not turn back on until reach back idle

        ''' ================================================ robot position ================================================ '''
        # Create a subscriber to the topic
        self.pos_subscription = self.create_subscription(
            Pose,
            'position',
            self.pos_callback,
            10)
        self.pos_y = self.pos_x = self.yaw = 0

        ''' ================================================ cmd_linear ================================================ '''
        # Create a publisher to the topic "cmd_linear", which can stop and move forward the robot
        self.linear_publisher = self.create_publisher(Int8, 'cmd_linear', 10)

        ''' ================================================ cmd_anglularVel ================================================ '''
        # Create a publisher to the topic "cmd_angle", which can rotate the robot
        self.anglularVel_publisher = self.create_publisher(Int8, 'cmd_anglularVel', 10)

        ''' ================================================ cmd_deltaAngle ================================================ '''
        # Create a publisher to the topic "cmd_angle", which can rotate the robot
        self.deltaAngle_publisher = self.create_publisher(Float64, 'cmd_deltaAngle', 10)

        ''' ================================================ robotControlNode_state_feedback ================================================ '''
        # Create a subscriber to the robotControlNode_state_feedback
        self.robotControl_subscription = self.create_subscription(
            String,
            'robotControlNode_state_feedback',
            self.robotControlNode_state_feedback_callback,
            10)
        
        # make this global so can be used to check moving or not
        self.linear_msg = Int8()

        ''' ================================================ boolCurve ================================================ '''
        # Create a publisher to the topic "boolCurve", which can changing bool for curve the robot
        self.boolCurve_publisher = self.create_publisher(Int8, 'boolCurve', 10)

        ''' ================================================ Master FSM ================================================ '''
        self.state = "idle"

        # used for navigation to jump back to the correct state afterwards,
        # if None then nothing to jump to
        self.magicState = "idle"

        # fsm_period = 0.1  # seconds
        fsm_period = 0.05  # seconds
        self.fsmTimer = self.create_timer(fsm_period, self.masterFSM)

        self.closestAngle = 0

        # Create a subscriber to the topic fsmDebug
        # to inject state changes for debugging in RQT
        self.fsmDebug_subscription = self.create_subscription(
            String,
            'fsmDebug',
            self.fsmDebug_callback,
            10)

        self.get_logger().info("MasterNode has started, bitchesss! >:D")

        # constants
        # self.linear_speed = 100
        # self.yaw_offset = 0
        self.recalc_freq = 10  # frequency to recalculate target angle and fix direction (10 means every one second)
        self.recalc_stat = 0

        self.path_recalc_freq = 20
        self.path_recalc_stat = 0

        self.dest_x = []
        self.dest_y = []
        self.path = []

        self.lastPlot = time.time()
        self.lastState = time.time()

        self.frontierPoints = []

        self.robotControlNodeState = ""

        self.lastPathUpdate = time.time()

        self.botx_pixel = 0
        self.boty_pixel = 0

        self.magicOriginx_pixel = 0
        self.magicOriginy_pixel = 0

        # for dijkstra
        # self.dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        # self.dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        self.dx = np.array([1, 0, -1, 0])
        self.dy = np.array([0, 1, 0, -1])
        self.d_row = []
        self.d_col = []
        self.d_data = []
        self.d_dim = (0, 0)
        # map values in the processed map (0 ~ 100) to evaluated values (1 ~ inf)
        self.d_cost = np.arange(101, dtype=np.float32)
        for i in range(101):
            if i <= 30:
                self.d_cost[i] = 1
            else:
                self.d_cost[i] = (71 / (101 - i) - 1) * 1e8 + 1
                
        self.bucketStarted = 0
        
        self.mustVisitPoints_m = MUST_VISIT_POINTS_M
        
        self.mustVisitPointsChecked_pixel = []
        
        self.frontierSkipThreshold = FRONTIER_SKIP_THRESHOLD

    def http_listener_callback(self, msg):
        # "idle", "door1", "door2", "connection error", "http error"
        self.doorStatus = msg.data

    def switch_listener_callback(self, msg):
        # "released" or "pressed"
        self.switchStatus = msg.data
        if self.state == "moving_to_bucket" and self.switchStatus == "pressed":
            self.state = self.magicState = "releasing"

            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)

            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)

            # go back to idle after releasing
            self.lastState = time.time()

    def scan_callback(self, msg):
        # create numpy array to store lidar data
        self.laser_range = np.array(msg.ranges)

        # # read min and max range values
        # self.range_min = msg.range_min
        # self.range_max = msg.range_max

        # self.get_logger().info("range_min: $s, range_max: $s" % (str(self.range_min), str(self.range_max)))

        # # replace out of range values with nan
        # self.laser_range[self.laser_range < self.range_min] = np.nan
        # self.laser_range[self.laser_range > self.range_max] = np.nan

        # replace 0's with nan
        self.laser_range[self.laser_range == 0] = np.nan

        # store the len since it changes
        self.range_len = len(self.laser_range)
        # self.get_logger().info(str(self.laser_range))

        self.bucketFrontLeftIndex = self.angle_to_index(BUCKET_FRONT_LEFT_ANGLE, self.range_len)
        self.bucketFrontRightIndex = self.angle_to_index(BUCKET_FRONT_RIGHT_ANGLE, self.range_len)

        self.leftIndexL = self.angle_to_index(LEFT_UPPER_ANGLE, self.range_len)
        self.leftIndexH = self.angle_to_index(LEFT_LOWER_ANGLE, self.range_len)

        self.rightIndexL = self.angle_to_index(RIGHT_LOWER_ANGLE, self.range_len)
        self.rightIndexH = self.angle_to_index(RIGHT_UPPER_ANGLE, self.range_len)

        self.backIndexL = self.angle_to_index(BACK_LOWER_ANGLE, self.range_len)
        self.backIndexH = self.angle_to_index(BACK_UPPER_ANGLE, self.range_len)

        self.mazeFrontLeftindex = self.angle_to_index(MAZE_FRONT_LEFT_ANGLE, self.range_len)
        self.mazeFrontRightindex = self.angle_to_index(MAZE_FRONT_RIGHT_ANGLE, self.range_len)

    def bucketAngle_listener_callback(self, msg):
        # # only update if everything is far away enough, this acts as a filter to prevent wrong angle when its too close to te bucket
        
        # if any(self.laser_range < BUCKET_TOO_CLOSE):
        #     self.get_logger().info('too close, bucket angle not updated')
        #     return
        # else:
        #     # shift down values and add newest to front
        #     self.bucketArray = [msg.data] + self.bucketArray[:-1]
            
        #     # medium of 5 values
        #     self.bucketAngle = np.median([msg.data])      
            
        #     self.get_logger().info('updated self.bucketAngle to %s' % str(self.bucketAngle))
        
        self.bucketAngle = msg.data
            
    def occ_callback(self, msg):
        occTime = time.time()
        self.get_logger().info('[occ_callback]: new occ map!')

        self.map_res = msg.info.resolution  # according to experiment, should be 0.05 m
        self.map_w = msg.info.width
        self.map_h = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        # this gives the locations of bot in the occupancy map, in pixel
        self.botx_pixel = round((self.pos_x - self.map_origin_x) / self.map_res)
        self.boty_pixel = round((self.pos_y - self.map_origin_y) / self.map_res)

        # this gives the locations of magic origin in the occupancy map, in pixel
        self.offset_x = round((-self.map_origin_x) / self.map_res) - self.magicOriginx_pixel
        self.offset_y = round((-self.map_origin_y) / self.map_res) - self.magicOriginy_pixel
        self.magicOriginx_pixel += self.offset_x
        self.magicOriginy_pixel += self.offset_y

        # calculate door and finish line coords in pixels, this may exceed the current occ map size since it extends beyong the explored area
        self.leftDoor_pixel = (round((LEFT_DOOR_COORDS_M[0] - self.map_origin_x) / self.map_res) + self.offset_x, round((LEFT_DOOR_COORDS_M[1] - self.map_origin_y) / self.map_res) + self.offset_y)
        self.rightDoor_pixel = (round((RIGHT_DOOR_COORDS_M[0] - self.map_origin_x) / self.map_res) + self.offset_x, round((RIGHT_DOOR_COORDS_M[1] - self.map_origin_y) / self.map_res) + self.offset_y)
        self.finishLine_pixel = (round((FINISH_LINE_M[0] - self.map_origin_x) / self.map_res) + self.offset_x, round((FINISH_LINE_M[1] - self.map_origin_y) / self.map_res) + self.offset_y)
        
        # MUST_VISIT_POINTS_M
        self.mustVisitPoints_pixel = [(round((x - self.map_origin_x) / self.map_res) + self.offset_x, round((y - self.map_origin_y) / self.map_res) + self.offset_y) for x, y in self.mustVisitPoints_m]
        
        # to avoid dicks frontier
        self.mazeBotBoundary = self.magicOriginy_pixel - 5
        self.mazeLeftBoundary = self.magicOriginx_pixel - 5
        self.mazeTopBoundary = self.finishLine_pixel[1]
        self.mazeRightBoundary = self.magicOriginx_pixel + 68
        
        # only apply shift to destination if OCC is disable
        # this means dest is still correct, just that map is not
        # if OCC is enable, then dest is corrected with shift below when new path is calculated
        if self.disableOCC == True:
            # add the offset to the all destination
            self.dest_x = [x + self.offset_x for x in self.dest_x]
            self.dest_y = [y + self.offset_y for y in self.dest_y]
        
        if self.disableOCC == False:
            self.get_logger().info('[occ_callback]: occ enabled')
            
            # ensure its stopped
            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)

            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)
            
            # disable occCallback
            self.disableOCC = True
            
            # set newOccFlag to True
            self.newOccFlag = True
            
            self.get_logger().info('[occ_callback]: setting newOccFlag: %s, disableOCC: %s' % (str(self.newOccFlag), str(self.disableOCC)))
            
            # self.get_logger().info('[occ_callback]: occ_callback took: %s' % timeTaken)
            #
            # TEMPPP
            # Convert the OccupancyGrid to a numpy array
            self.oriorimap = np.array(msg.data, dtype=np.float32).reshape(msg.info.height, msg.info.width)

            # hardcode
            for i in range(-3, 9):
                ny = self.magicOriginy_pixel + 4
                nx = self.magicOriginx_pixel + i
                if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
                    self.oriorimap[ny][nx] = 100

            for i in range(-3, 70):
                ny = self.magicOriginy_pixel - 4
                nx = self.magicOriginx_pixel + i
                if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
                    self.oriorimap[ny][nx] = 100

            for i in range(-4, 41):
                ny = self.magicOriginy_pixel + i
                nx = self.magicOriginx_pixel - 4
                if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
                    self.oriorimap[ny][nx] = 100

            for i in range(-4, 41):
                ny = self.magicOriginy_pixel + i
                nx = self.magicOriginx_pixel + 70
                if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
                    self.oriorimap[ny][nx] = 100

            # for i in range(-3, 24):
            #     ny = self.magicOriginy_pixel + 40
            #     nx = self.magicOriginx_pixel + i
            #     if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
            #         self.oriorimap[ny][nx] = 100
            #
            # for i in range(40, 70):
            #     ny = self.magicOriginy_pixel + 40
            #     nx = self.magicOriginx_pixel + i
            #     if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
            #         self.oriorimap[ny][nx] = 100

            # this converts the occupancy grid to an 1d array of map, umpapped, occupied
            occ_counts, edges, binnum = scipy.stats.binned_statistic(np.array(msg.data), np.nan, statistic='count',
                                                                    bins=occ_bins)

            # reshape to 2D array
            # 1 = unmapped
            # 2 = mapped and open
            # 3 = mapped and obstacle
            self.occupancyMap = np.uint8(binnum.reshape(msg.info.height, msg.info.width))
            
            # then convert to grid pixel by dividing map_res in m/cell, +0.5 to round up
            # pixelExpend = numbers of pixel to expend by
            pixelExpend = math.ceil(CLEARANCE_RADIUS / (self.map_res * 100))

            # Create a mask of the UNMAPPED areas
            unmapped_mask = (self.occupancyMap == UNMAPPED)

            # Create a mask of the OPEN areas
            open_mask = (self.occupancyMap == OPEN)

            # Create a mask of the OBSTACLE areas
            obstacle_mask = (self.occupancyMap == OBSTACLE)

            # Create a structuring element for the dilation
            selem = disk(pixelExpend)

            # Perform the dilation
            dilated = dilation(obstacle_mask, selem)

            # Apply the dilation only within the OPEN areas
            self.dilutedOccupancyMap = np.where((dilated & open_mask), OBSTACLE, self.occupancyMap)
            
            # check if must visit poiints are in the map if yes then:
            # if must visit points are not unmapped, then remove them
            self.mustVisitPointsChecked_pixel = [(x, y) for x, y in self.mustVisitPoints_pixel if not (0 < x < self.map_w and 0 < y < self.map_h and (self.dilutedOccupancyMap[y][x] == OPEN or self.dilutedOccupancyMap[y][x] == OBSTACLE))]
            
            # Normalize the values to the range [0, 1]
            # self.oriorimap /= 100.0

            # # Print all unique values in self.oriorimap
            # unique_values = np.unique(self.oriorimap)
            # self.get_logger().info('Unique values in oriorimap: %s' % unique_values)

            # Define the function to apply over the moving window
            def func(window):
                # Calculate the distances from the center of the grid
                center = WINDOWSIZE // 2
                distances = np.sqrt(
                    (np.arange(WINDOWSIZE) - center) ** 2 + (np.arange(WINDOWSIZE)[:, None] - center) ** 2).reshape(
                    WINDOWSIZE ** 2)
                distances *= 2.5  # TEMP

                # Calculate the new pixel value
                new_pixel = np.max(window * PARAMETER_R ** distances)

                return new_pixel

            # Apply the function over a moving window on the image
            self.processedOcc = np.round(generic_filter(self.oriorimap, func, size=(WINDOWSIZE, WINDOWSIZE))).astype(int)

            # regard unmapped area as perfectly blocked area
            self.processedOcc[unmapped_mask] = 100

            # self.get_logger().info(str(self.processedOcc == self.oriorimap))

            self.dijkstra()

            # find frontier points
            self.frontierSearch()

            self.path_recalc_stat += 1
            self.path_recalc_stat %= self.path_recalc_freq
            if len(self.dest_x) > 0:
                # if self.magicState == "frontier_search" and (
                # self.dest_y[-1] + self.offset_y, self.dest_x[-1] + self.offset_x) not in self.frontier:
                #     # set linear to be zero
                #     linear_msg = Int8()
                #     linear_msg.data = 0
                #     self.linear_publisher.publish(linear_msg)

                #     # set delta angle = 0 to stop
                #     deltaAngle_msg = Float64()
                #     deltaAngle_msg.data = 0.0
                #     self.deltaAngle_publisher.publish(deltaAngle_msg)

                #     self.dest_x = []
                #     self.dest_y = []

                #     self.get_logger().info('[occ_callback]: designation is already mapped; get back to frontier_search')

                #     self.state = self.magicState

                #     timeTaken = time.time() - occTime
                #     self.get_logger().info('[occ_callback]: occ_callback took: %s' % timeTaken)
                #     return
                
                if self.magicState == "frontier_search" and self.dist[self.dest_y[-1] + self.offset_y][self.dest_x[-1] + self.offset_x] > self.frontierSkipThreshold:
                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)

                    # set delta angle = 0 to stop
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = 0.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                    
                    self.dest_x.clear()
                    self.dest_y.clear()
                    self.get_logger().info('[occ_callback]: no path found get back to magicState: %s' % self.magicState)
                    self.state = self.magicState
                    return
                
                if self.path_recalc_stat != 0:
                    timeTaken = time.time() - occTime
                    self.get_logger().info('[occ_callback]: occ_callback took: %s' % timeTaken)
                    return

                # check the path for the last point which is the destination set last time
                new_dest_x, new_dest_y = self.find_path_to(self.dest_x[-1] + self.offset_x, self.dest_y[-1] + self.offset_y)

                if len(new_dest_x) == 0:
                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)

                    # set delta angle = 0 to stop
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = 0.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                    
                    self.dest_x.clear()
                    self.dest_y.clear()
                    self.get_logger().info('[occ_callback]: no path found get back to magicState: %s' % self.magicState)
                    self.state = self.magicState
                    return

                # remove the current position which lies at the front of array
                if len(new_dest_x) > 1:
                    new_dest_x = new_dest_x[1:]
                    new_dest_y = new_dest_y[1:]

                # compare the new path with the old path
                if new_dest_x != self.dest_x or new_dest_y != self.dest_y:
                    self.get_logger().info('[occ_callback]: path updated')
                    # if the first target point changes, stop once and move again
                    if new_dest_x[0] != self.dest_x[0] or new_dest_y[0] != self.dest_y[0]:
                        # set linear to be zero
                        self.linear_msg.data = 0
                        self.linear_publisher.publish(self.linear_msg)

                        # set delta angle = 0 to stop
                        deltaAngle_msg = Float64()
                        deltaAngle_msg.data = 0.0
                        self.deltaAngle_publisher.publish(deltaAngle_msg)

                        self.move_straight_to(new_dest_x[0], new_dest_y[0])
                    # update target points
                    self.dest_x = new_dest_x
                    self.dest_y = new_dest_y

            timeTaken = time.time() - occTime
            self.get_logger().info('[occ_callback]: occ_callback took: %s' % timeTaken)
            
    def pos_callback(self, msg):
        # Note: those values are different from the values obtained from odom
        self.pos_x = msg.position.x
        self.pos_y = msg.position.y
        # in degrees (not radians)
        self.yaw = angle_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        # self.yaw += self.yaw_offset
        # self.get_logger().info('x y yaw: %f %f %f' % (self.pos_x, self.pos_y, self.yaw))

    def robotControlNode_state_feedback_callback(self, msg):
        self.robotControlNodeState = msg.data

    def fsmDebug_callback(self, msg):
        self.stateList = ["idle", 
                          "maze_rotating", 
                          "maze_moving", 
                        #   "escape_wall_lmao", 
                        #   "naive_frontier_finding_point",
                        #   "naive_frontier_checking",
                        #   "naive_front_rotating_to_frontier",
                        #   "naive_front_moving_to_frontier",
                        #   "naive_back_rotating_to_frontier",
                        #   "naive_back_moving_to_frontier",
                          "frontier_search",
                          "visiting_must_visit",
                          "move_to_hallway", 
                          "http_request", 
                          "go_to_left_door", 
                          "go_to_right_door", 
                          "rotate_to_left_door", 
                          "rotate_to_right_door", 
                          "enter_to_left_door",
                          "enter_to_right_door",
                          "checking_walls_distance"] 
        if msg.data in self.stateList:
            self.state = self.magicState = msg.data
            
            if msg.data == "checking_walls_distance":
                # only used for injecting 
                # set boolCurve to 1, to be place in the state before checking_walls_distance
                boolCurve_msg = Int8()
                boolCurve_msg.data = 1
                self.boolCurve_publisher.publish(boolCurve_msg)
                
                self.disableOCC = True
                
        else:
            mode, tx, ty = map(int, self.state.split())
            if mode == 0:
                self.dest_x.append(tx)
                self.dest_y.append(ty)
                self.move_straight_to(tx, ty)
            elif mode == 1:
                self.move_to(tx, ty)
            else:
                self.get_logger().info('mode %d does not exist' % mode)
    
    def index_to_angle(self, index, arrLen):
        # return in degrees
        return (index / (arrLen - 1)) * 359

    def angle_to_index(self, angle, arrLen):
        # take deg give index
        return int((angle / 359) * (arrLen - 1))

    def custom_destroy_node(self):
        # set linear to be zero
        self.linear_msg.data = 0
        self.linear_publisher.publish(self.linear_msg)

        # set delta angle = 0 to stop
        deltaAngle_msg = Float64()
        deltaAngle_msg.data = 0.0
        self.deltaAngle_publisher.publish(deltaAngle_msg)

        self.destroy_node()

    def masterFSM(self):
        # testing lidar
        # if any(self.laser_range[:self.mazeFrontLeftindex] < NAV_TOO_CLOSE) \
        #             or any(self.laser_range[self.mazeFrontRightindex:] < NAV_TOO_CLOSE) \
        #             or np.all(np.isnan(self.laser_range[:self.mazeFrontLeftindex])) \
        #             or np.all(np.isnan(self.laser_range[:self.mazeFrontLeftindex])):
        #         self.get_logger().warn('[maze_moving]: ahhh wall to close to front uwu')

        # if not any(self.laser_range[self.backIndexL:self.backIndexH] < NAV_TOO_CLOSE) \
        #     and not np.all(np.isnan(self.laser_range[self.backIndexL:self.backIndexH])):
        #     self.get_logger().warn('[maze_moving]: back is clear')
        
        # return
            
        self.get_logger().info('[masterFSM]: self.state: %s, self.magicState %s' % (self.state, self.magicState))

        # if no more frontier and its in frontier search means done and go to visiting_must_visit
        if self.magicState == "frontier_search" and self.frontierPoints == []:
            self.get_logger().info('[masterFSM]: no more frontier points go to visiting_must_visit')
            self.state = self.magicState = "visiting_must_visit"
            
            # set to stop since it sometimes hit to wall during state transistion, as there is not obstacle avoidance outside of moving
            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)

            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)
            
            # call for new map again before going visiting_must_visit so the must visit points are updated
            # enable occCallback
            self.disableOCC = False
            
            # set newOccFlag to False
            self.newOccFlag = False
            
            self.visiting_must_visit_start = time.time()
            
        # if no more must visit points and its in visiting_must_visit means really done so go to wait_for_frontier
        if self.magicState == "visiting_must_visit" and self.frontierPoints == [] and (time.time() - self.visiting_must_visit_start) > 5:
            self.get_logger().info('[masterFSM]: no more points to visit go to wait_for_frontier')
            self.state = self.magicState = "wait_for_frontier"
            self.lastWaitForFrontier = time.time()
            
            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)

            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)
            
            # enable occCallback
            self.disableOCC = False
            
            # set newOccFlag to False
            self.newOccFlag = False
            
        # get rid cuz have visiting_must_visit now 
        # # if finding hallway but cutting through wall, and frontier pops up, then go back to frontier search
        # if self.magicState == "move_to_hallway" and self.frontierPoints != []:
        #     self.get_logger().info('[masterFSM]: frontier points found while moving to hallway, go back to frontier search')
        #     self.state = self.magicState = "frontier_search"
                
        listStateIgnoreForObstacle = ["idle", 
                                      "enter_to_left_door",
                                      "enter_to_right_door",
                                      "checking_walls_distance",
                                      "rotating_to_move_away_from_walls",
                                      "rotating_to_bucket",
                                      "moving_to_bucket",
                                      "releasing",
                                      ]
                
        # special case for maze_rotating
        # if moving then need to check for obstacle, else rotate on spot no need to check
        if ((self.state not in listStateIgnoreForObstacle) or (self.state == "maze_rotating" and self.linear_msg.data != 0)) and self.linear_msg.data != 0:
            self.get_logger().warn('[masterFSM]: self.state = %s, self.linear_msg.data = %d' % (self.state, self.linear_msg.data))
            
            frontNotClearNav = any(self.laser_range[:self.mazeFrontLeftindex] < NAV_TOO_CLOSE) \
                                or any(self.laser_range[self.mazeFrontRightindex:] < NAV_TOO_CLOSE) \
                                or np.all(np.isnan(self.laser_range[:self.mazeFrontLeftindex])) \
                                or np.all(np.isnan(self.laser_range[self.mazeFrontRightindex:]))
            
            buttNotClearNav = any(self.laser_range[self.backIndexL:self.backIndexH] < NAV_TOO_CLOSE) \
                                or np.all(np.isnan(self.laser_range[self.backIndexL:self.backIndexH]))
            
            # if obstacle in front and close to both sides, rotate to move between the two
            if frontNotClearNav:
                self.get_logger().warn('[masterFSM]: ahhh wall to close to front uwu')
                
                # stop
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)

                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)
                
                time.sleep(0.2)
                
                # if both front and back are close, rotate to the clearest angle in front (which is hopefully the initial direction)
                # else do the reverse and rotate thing
                
                if frontNotClearNav and buttNotClearNav: 
                    anglularVel_msg = Int8()
                    
                    if np.nanmin(self.laser_range[:self.mazeFrontLeftindex]) < np.nanmin(self.laser_range[self.mazeFrontRightindex:]):
                        anglularVel_msg.data = -100
                    else:
                        anglularVel_msg.data = 100
                        
                    self.anglularVel_publisher.publish(anglularVel_msg)
                else:
                    # rotate to face obstacle
                    anglularVel_msg = Int8()
                    
                    self.rotateBack = 0
                    
                    if np.nanmin(self.laser_range[:self.mazeFrontLeftindex]) < np.nanmin(self.laser_range[self.mazeFrontRightindex:]):
                        anglularVel_msg.data = 100
                        self.rotateBack = -100
                    else:
                        anglularVel_msg.data = -100
                        self.rotateBack = 100
                        
                    self.anglularVel_publisher.publish(anglularVel_msg)
                    
                    start = time.time()
                    while (time.time() - start < 0.5):
                        pass
                    
                    anglularVel_msg.data = 0
                    self.anglularVel_publisher.publish(anglularVel_msg)
                    
                    time.sleep(0.2)
                    
                    # reverse
                    self.linear_msg.data = int(-LIN_MAX / 3.2)
                    self.linear_publisher.publish(self.linear_msg)
                    start = time.time()
                    while (time.time() - start < 0.4 * 1.6):
                        pass
                    
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)
                    
                    time.sleep(0.2)
                    
                    # rotate back to face front
                    anglularVel_msg = Int8()
                    anglularVel_msg.data = self.rotateBack
                        
                    self.anglularVel_publisher.publish(anglularVel_msg)
                    
                    start = time.time()
                    while (time.time() - start < 0.5):
                        pass
                    
                    anglularVel_msg.data = 0
                    self.anglularVel_publisher.publish(anglularVel_msg)
                    
                    # ori below

                    # # set linear to be zero
                    # self.linear_msg.data = 0
                    # self.linear_publisher.publish(self.linear_msg)

                    # # set delta angle = 0 to stop
                    # deltaAngle_msg = Float64()
                    # deltaAngle_msg.data = 0.0
                    # self.deltaAngle_publisher.publish(deltaAngle_msg)
                    
                    # time.sleep(0.5)
                    
                    # # move backward for 0.5 sec as long as butt has space
                    # self.linear_msg.data = int(-LIN_MAX / 1.5)
                    # self.linear_publisher.publish(self.linear_msg)
                    # start = time.time()
                    # while (time.time() - start < 0.4*1.5) \
                    #     and not any(self.laser_range[self.backIndexL:self.backIndexH] < NAV_TOO_CLOSE) \
                    #     and not np.all(np.isnan(self.laser_range[self.backIndexL:self.backIndexH])):
                    #     pass
                    
                    # self.linear_msg.data = 0
                    # self.linear_publisher.publish(self.linear_msg)
                    
                    # anglularVel_msg = Int8()
                    
                    # # # rotate to next point if have next point else just rotate to away from the close one
                    # # if len(self.dest_y) > 1:
                    # #     target_yaw = math.atan2(self.dest_y[0] - self.boty_pixel, self.dest_x[0] - self.botx_pixel) * (
                    # #             180 / math.pi)

                    # #     deltaAngle = target_yaw - self.yaw

                    # #     self.get_logger().info('[masterFSM]: align to next dest: %f' % deltaAngle)

                    # #     self.linear_msg.data = 0
                    # #     self.linear_publisher.publish(self.linear_msg)

                    # #     # set delta angle to rotate to target angle
                    # #     deltaAngle_msg = Float64()
                    # #     deltaAngle_msg.data = deltaAngle * 1.0
                    # #     self.deltaAngle_publisher.publish(deltaAngle_msg)

                    # #     self.state = "maze_rotating"
                    
                    # # else:
                    # self.get_logger().info('[masterFSM]: rotate away from close one')
                    
                    # # rotate to away from the close one for 0.5 sec
                    # if np.nanmean(self.laser_range[:self.mazeFrontLeftindex]) < np.nanmean(self.laser_range[self.mazeFrontRightindex:]):
                    #     anglularVel_msg.data = -120
                    # else:
                    #     anglularVel_msg.data = 120
                        
                    # self.anglularVel_publisher.publish(anglularVel_msg)
                    
                    # start = time.time()
                    # while (time.time() - start < 1):
                    #     pass
                
                    # anglularVel_msg.data = 0
                    # self.anglularVel_publisher.publish(anglularVel_msg)
                    
                    # # move forward for 0.5 sec and infront is nothing close
                    # self.linear_msg.data = int(LIN_MAX / 1.5)
                    # self.linear_publisher.publish(self.linear_msg)
                    # start = time.time()
                    # while (time.time() - start < 0.5 * 1.5) \
                    #     and not any(self.laser_range[:self.mazeFrontLeftindex] < NAV_TOO_CLOSE) \
                    #     and not any(self.laser_range[self.mazeFrontRightindex:] < NAV_TOO_CLOSE) \
                    #     and not np.all(np.isnan(self.laser_range[:self.mazeFrontLeftindex])) \
                    #     and not np.all(np.isnan(self.laser_range[self.mazeFrontRightindex:])):
                    #     pass
                
                # get rid of point that is too close to wall in the first place and take the next one
                # cannot take final one if its like thru a wall
                self.dest_x.clear()
                self.dest_y.clear()
                
                self.get_logger().warn(
                    '[masterFSM]: get back to magicState: %s' % self.magicState)
                self.state = self.magicState
                
                # enable occCallback
                self.disableOCC = False
                
                # set newOccFlag to False
                self.newOccFlag = False
                
                self.get_logger().info('[masterFSM]: setting newOccFlag: %s, disableOCC: %s' % (str(self.newOccFlag), str(self.disableOCC)))
                
        # if occ is enabled, means new values are wanted
        # so if newOccFlag is false, then need to wait until newOccFlag is true before procedding with whatever state it is in
        if self.newOccFlag == False and self.disableOCC == False:
            self.get_logger().info('[masterFSM]: waiting for new occ map, newOccFlag: %s, disableOCC: %s' % (str(self.newOccFlag), str(self.disableOCC)))
            
            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)

            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)
            
            return
        else:
            self.get_logger().info('[masterFSM]: new occ map received can run')
                
        if self.state == "idle":
            # reset servo to 90, to block ballsssss
            servoAngle_msg = UInt8()
            servoAngle_msg.data = 90
            self.servo_publisher.publish(servoAngle_msg)

            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)

            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)

            # off limit switch
            switch_msg = String()
            switch_msg.data = "deactivate"
            self.switch_publisher.publish(switch_msg)

            # set boolCurve to 1
            boolCurve_msg = Int8()
            boolCurve_msg.data = 1
            self.boolCurve_publisher.publish(boolCurve_msg)
            
            # reset to not disable OCC callback and newOccFlag
            self.disableOCC = False
            self.newOccFlag = False

        elif self.state == "maze_rotating":
             # self.get_logger().info('current yaw: %f' % self.yaw)
            
            self.get_logger().info('[maze_rotating]: rotating')
            
            if self.robotControlNodeState == "rotateStop":
                # check that dist is not empty (its empty for cases where maze_moving is used to rotate only)
                if len(self.dest_x) == 0:
                    self.get_logger().info('[maze_rotating]: no more destination; get back to magicState: %s' % self.magicState)
                    self.state = self.magicState
                    return
                
                # set linear to start moving forward
                self.linear_msg.data = LIN_MAX
                self.linear_publisher.publish(self.linear_msg)
                
                self.state = "maze_moving"
                
                # reset recalc_stat
                self.recalc_stat = 0
        elif self.state == "maze_moving":                
            # if reached the destination (within one pixel), stop and move to the next destination
            self.get_logger().info('[maze_moving]: moving')

            if abs(self.botx_pixel - self.dest_x[0]) <= PIXEL_DEST_THRES and abs(
                    self.boty_pixel - self.dest_y[0]) <= PIXEL_DEST_THRES:
                
                # to recover from overshooting
                curr_pos = np.array([self.botx_pixel, self.boty_pixel])
                x_coords = np.array(self.dest_x)
                y_coords = np.array(self.dest_y)
                
                # Combine the x and y coordinates into a list of points
                points = np.vstack((x_coords, y_coords)).T

                # Calculate the Euclidean distance between the point and each point in the list
                distances = np.linalg.norm(points - curr_pos, axis=1)

                # Find the index of the point with the smallest distance
                indexCut = np.argmin(distances)
                
                # jump to the closest point
                self.dest_x = self.dest_x[indexCut:]
                self.dest_y = self.dest_y[indexCut:]
                
                self.get_logger().info('[maze_moving]: finished moving')

                self.dest_x = self.dest_x[1:]
                self.dest_y = self.dest_y[1:]

                if len(self.dest_x) == 0:
                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)

                    # set delta angle = 0 to stop
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = 0.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                    
                    self.get_logger().info(
                        '[maze_moving]: no more destination; get back to magicState: %s and enable occCallback' % self.magicState)
                    
                    # enable occCallback
                    self.disableOCC = False
                    
                    # set newOccFlag to False
                    self.newOccFlag = False
                    
                    self.get_logger().info('[maze_moving]: setting newOccFlag: %s, disableOCC: %s' % (str(self.newOccFlag), str(self.disableOCC)))
                    
                    self.state = self.magicState
                else:
                    # this will aim to next point
                    self.move_straight_to(self.dest_x[0], self.dest_y[0])
                return

            # # if obstacle in front and close to both sides, rotate to move between the two
            # if any(self.laser_range[:self.mazeFrontLeftindex] < NAV_TOO_CLOSE) \
            #         or any(self.laser_range[self.mazeFrontRightindex:] < NAV_TOO_CLOSE) \
            #         or np.all(np.isnan(self.laser_range[:self.mazeFrontLeftindex])) \
            #         or np.all(np.isnan(self.laser_range[self.mazeFrontRightindex:])):
            #     self.get_logger().warn('[maze_moving]: ahhh wall to close to front uwu')

            #     # set linear to be zero
            #     linear_msg = Int8()
            #     linear_msg.data = 0
            #     self.linear_publisher.publish(linear_msg)

            #     # set delta angle = 0 to stop
            #     deltaAngle_msg = Float64()
            #     deltaAngle_msg.data = 0.0
            #     self.deltaAngle_publisher.publish(deltaAngle_msg)
                
            #     # move backward for 0.5 sec as long as butt has space
            #     linear_msg.data = -self.linear_speed
            #     self.linear_publisher.publish(linear_msg)
            #     start = time.time()
            #     while (time.time() - start < 0.5) \
            #         and not any(self.laser_range[self.backIndexL:self.backIndexH] < NAV_TOO_CLOSE) \
            #         and not np.all(np.isnan(self.laser_range[self.backIndexL:self.backIndexH])):
            #         pass
                
            #     linear_msg.data = 0
            #     self.linear_publisher.publish(linear_msg)
                
            #     anglularVel_msg = Int8()
                
            #     # rotate to away from the close one for 0.5 sec
            #     if np.nanmean(self.laser_range[:self.mazeFrontLeftindex]) < np.nanmean(self.laser_range[self.mazeFrontRightindex:]):
            #         anglularVel_msg.data = -100
            #     else:
            #         anglularVel_msg.data = 100
                    
            #     self.anglularVel_publisher.publish(anglularVel_msg)
                
            #     start = time.time()
            #     while (time.time() - start < 0.5):
            #         pass
                
            #     anglularVel_msg.data = 0
            #     self.anglularVel_publisher.publish(anglularVel_msg)

            #     # get rid of point that is too close to wall in the first place and take the next one
            #     # cannot take final one if its like thru a wall
            #     self.dest_x.clear()
            #     self.dest_y.clear()
                
            #     self.get_logger().warn(
            #         '[maze_moving]: get back to magicState: %s' % self.magicState)
            #     self.state = self.magicState
            #     return

            # else:
            # else just increment counter for re orient
            self.recalc_stat += 1

            # recalculate target angle if reach recalc_freq
            # this takes care both for obstacles and re aiming to target coords
            if self.recalc_stat == self.recalc_freq:
                self.get_logger().info('[maze_moving]: recalc')

                self.recalc_stat = 0

                target_yaw = math.atan2(self.dest_y[0] - self.boty_pixel, self.dest_x[0] - self.botx_pixel) * (
                        180 / math.pi)

                deltaAngle = target_yaw - self.yaw

                self.get_logger().info('[maze_moving]: front open, reallign with deltaAngle: %f' % deltaAngle)
                
                
                # if deltaAnfle is too small, just ignore
                # if deltaAngle is too big, stop then rotate
                # else, rotate and move at the same time
                if abs(deltaAngle) < 15:
                    pass
                elif abs(deltaAngle) >= 45:
                    # set linear
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)
                    
                    # wait abit first
                    time.sleep(0.5)

                    # set delta angle to rotate to target angle
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = deltaAngle * 1.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)

                    self.state = "maze_rotating"
                else:
                    # set linear
                    self.linear_msg.data = LIN_WHEN_ROTATING
                    self.linear_publisher.publish(self.linear_msg)

                    # set delta angle to rotate to target angle
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = deltaAngle * 1.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)

                    self.state = "maze_rotating"
                
        # elif self.state == "escape_wall_lmao":
        #     # dequeue all bullshit in dest
        #     self.dest_x = []
        #     self.dest_y = []

        #     # Find the nearest free pixel
        #     free_pixels = np.where(self.dilutedOccupancyMap == OPEN)
        #     distances = np.sqrt((free_pixels[0] - self.boty_pixel) ** 2 + (free_pixels[1] - self.botx_pixel) ** 2)
        #     minIndex = np.argmin(distances)

        #     # cannot use move_to since its stuck in a wall
        #     # use move_straight_to instead
        #     self.dest_x = [free_pixels[1][minIndex]]
        #     self.dest_y = [free_pixels[0][minIndex]]
        #     self.get_logger().info(
        #         '[escape_wall_lmao]: currently at (%d, %d) moving to nearest free pixel: (%d, %d)' % (
        #         self.botx_pixel, self.boty_pixel, self.dest_x[0], self.dest_y[0]))

        #     # if free pixel is behind (90, 270), reverse first
        #     # else use move_straight_to to move to the free pixel
        #     target_yaw = math.atan2(self.dest_y[0] - self.boty_pixel, self.dest_x[0] - self.botx_pixel) * (
        #                 180 / math.pi)
        #     deltaAngle = target_yaw - self.yaw

        #     if deltaAngle > 90 or deltaAngle < -90:
        #         # set linear to be reverse
        #         linear_msg = Int8()
        #         linear_msg.data = -self.linear_speed
        #         self.linear_publisher.publish(linear_msg)

        #         # set delta angle to 0
        #         deltaAngle_msg = Float64()
        #         deltaAngle_msg.data = 0.0
        #         self.deltaAngle_publisher.publish(deltaAngle_msg)

        #         # set to maze_moving to move until the free pixel
        #         self.state = "maze_moving"
        #     else:
        #         self.move_straight_to(self.dest_x[0], self.dest_y[0])

        elif self.state == "frontier_search" or self.state == "visiting_must_visit":
            # # compare two frontier points and judge which we go first
            # # return True if p1 has higher priority than p2
            # def cmp(p1, p2):
            #     return p1[0] < p2[0]

            # destination = self.frontierPoints[0]
            # for i in range(1, len(self.frontierPoints)):
            #     if cmp(self.frontierPoints[i], destination):
            #         destination = self.frontierPoints[i]

            # Find the point in self.frontierPoints that is closest to the current position
            
            if len(self.frontierPoints) > 0:
                destination = self.frontierPoints[0]

                self.get_logger().info('[frontier_search]: next destination: (%d, %d)' % (destination[0], destination[1]))

                self.move_to(destination[0], destination[1])
                
                # visiting_must_visit is for:
                # for must visit points that had not been visited, it will be added to fronieterPoints by the frontierSearch functions
                # frontierSearch functions check if the state is visiting_must_visit before adding must visit to frontierPoints
                # mean while if there is any other actual frontier points, it will be added to frontierPoints as well
                # once all must visit points are visited, it will go to wait_for_frontier
            
        elif self.state == "wait_for_frontier":
            # after WAIT_FRONTIER sec, go to move_to_hallway
            
            self.get_logger().info('[wait_for_frontier]: waitng until %f, now is %f' % (WAIT_FRONTIER, time.time() - self.lastWaitForFrontier))
            
            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)

            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)
            
            # enable occCallback
            self.disableOCC = False
            
            # set newOccFlag to False
            self.newOccFlag = False
            
            if len(self.frontierPoints) > 0:
                self.get_logger().info('[wait_for_frontier]: frontier points found go to frontier_search')
                self.state = self.magicState = "frontier_search"
                return
            
            if time.time() - self.lastWaitForFrontier > WAIT_FRONTIER:
                self.get_logger().info('[wait_for_frontier]: time is up go to move_to_hallway')
                self.state = self.magicState = "move_to_hallway"
                
                # enable occCallback
                self.disableOCC = False
                
                # set newOccFlag to False
                self.newOccFlag = False
                return
            
        elif self.state == "move_to_hallway":
            # if reached hallway or is already in hall way, stop and start http_request
            if (abs(self.botx_pixel - self.finishLine_pixel[0]) <= PIXEL_DEST_THRES and abs(
                    self.boty_pixel - self.finishLine_pixel[1]) <= PIXEL_DEST_THRES) \
                or \
                (self.boty_pixel >= self.finishLine_pixel[1]):
                
                # set linear to be zero
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)

                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)

                self.get_logger().info('[move_to_hallway]: finished moving go http')
            
                # set magicState to be http_request, so that once at hall way point, it will open the door
                self.state = self.magicState = "http_request"
                                
                return
                
            # check if hallway is even in map or if its open in map, if its not, means frontier has not been fully explored, go back to frontier search
            # froniter search changes as confidence in map changes hence sometime when there is no more frontier, but after some time, there is more
            if self.finishLine_pixel[0] < 0 or self.finishLine_pixel[1] < 0 or self.finishLine_pixel[0] >= self.map_w or self.finishLine_pixel[1] >= self.map_h or self.dilutedOccupancyMap[self.finishLine_pixel[1], self.finishLine_pixel[0]] == UNMAPPED:
                self.state = self.magicState = "frontier_search"
                
                # make it more sensative to frontiers
                self.frontierSkipThreshold = self.frontierSkipThreshold * 0.9
                
                # enable occCallback
                self.disableOCC = False
                
                # set newOccFlag to False
                self.newOccFlag = False
                
                self.get_logger().warn('[move_to_hallway]: finishLine_pixel: (%d, %d) is not reachable is not in map, self.frontierSkipThreshold = %f' % (self.finishLine_pixel[0], self.finishLine_pixel[1], self.frontierSkipThreshold))
                
            else:
                # check if hall way is reachable
                # if not reachable, throw error (or do somthing else)
                # else move to the hallway
                if len(self.find_path_to(self.finishLine_pixel[0], self.finishLine_pixel[1])[0]) == 0:
                    self.get_logger().warn('[move_to_hallway]: finishLine_pixel: (%d, %d) is not reachable' % (self.finishLine_pixel[0], self.finishLine_pixel[1]))
                else:
                    self.get_logger().info('[move_to_hallway]: going to finishLine_pixel: (%d, %d)' % (self.finishLine_pixel[0], self.finishLine_pixel[1]))
                    self.move_to(self.finishLine_pixel[0], self.finishLine_pixel[1])

        elif self.state == "http_request":
            if self.doorStatus == "idle":
                # send openDoor request
                door_msg = String()
                door_msg.data = "openDoor"
                self.http_publisher.publish(door_msg)
                self.get_logger().info('[http_request]: opening door')

            elif self.doorStatus == "door1":
                self.get_logger().info('[http_request]: door1 opened')
                
                if WHITE_FLAG == False:
                    # if nav works
                    self.state = self.magicState = "go_to_left_door"
                    
                    # add a delay before read occ so that the door is fully opened
                    time.sleep(5)
                    
                    # enable occCallback
                    self.disableOCC = False
                    
                    # set newOccFlag to False
                    self.newOccFlag = False
                else:
                    # if nav is still not working, put bot at centre of both door facing right door and start http_request
                    self.state = self.magicState = "rotate_to_left_door"
                    
                    # add a delay before read occ so that the door is fully opened
                    time.sleep(5)

            elif self.doorStatus == "door2":
                self.get_logger().info('[http_request]: door2 opened')
                
                if WHITE_FLAG == False:
                    # if nav works
                    self.state = self.magicState = "go_to_right_door"
                    
                    # add a delay before read occ so that the door is fully opened
                    time.sleep(5)
                    
                    # enable occCallback
                    self.disableOCC = False
                    
                    # set newOccFlag to False
                    self.newOccFlag = False
                else:
                    # if nav is still not working, put bot at centre of both door facing right door and start http_request
                    self.state = self.magicState = "rotate_to_right_door"
                    
                    # add a delay before read occ so that the door is fully opened
                    time.sleep(5)

            elif self.doorStatus == "connection error":
                self.get_logger().warn('[http_request]: connection error')

            elif self.doorStatus == "http error":
                self.get_logger().warn('[http_request]: http error')

            else:
                self.get_logger().warn('[http_request]: msg error')

        elif self.state == "go_to_left_door":
            if abs(self.botx_pixel - self.leftDoor_pixel[0]) <= PIXEL_DEST_THRES and abs(
                    self.boty_pixel - self.leftDoor_pixel[1]) <= PIXEL_DEST_THRES:
                
                # set linear to be zero
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)

                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)

                self.get_logger().info('[go_to_left_door]: finished moving go rotate_to_left_door')
            
                # set magicState to be rotate_to_left_door, so that once at door, it will rotate_to_left_door
                self.state = self.magicState = "rotate_to_left_door"
                            
                return
            
            # check if left door is reachable
            # if not reachable, throw error (or do somthing else)
            # else move to the left door
            if len(self.find_path_to(self.leftDoor_pixel[0], self.leftDoor_pixel[1])[0]) == 0:
                self.get_logger().warn('[go_to_left_door]: leftDoor_pixel: (%d, %d) is not reachable' % (self.leftDoor_pixel[0], self.leftDoor_pixel[1]))
            else:
                self.get_logger().info('[go_to_left_door]: going to leftDoor_pixel: (%d, %d)' % (self.leftDoor_pixel[0], self.leftDoor_pixel[1]))
                self.move_to(self.leftDoor_pixel[0], self.leftDoor_pixel[1])
                
        elif self.state == "go_to_right_door":
            if abs(self.botx_pixel - self.rightDoor_pixel[0]) <= PIXEL_DEST_THRES and abs(
                    self.boty_pixel - self.rightDoor_pixel[1]) <= PIXEL_DEST_THRES:
                
                # set linear to be zero
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)

                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)

                self.get_logger().info('[go_to_right_door]: finished moving go rotate_to_right_door')
            
                # set magicState to be rotate_to_right_door, so that once at door, it will rotate_to_right_door
                self.state = self.magicState = "rotate_to_right_door"
          
                return
            
            # check if left door is reachable
            # if not reachable, throw error (or do somthing else)
            # else move to the left door
            if len(self.find_path_to(self.rightDoor_pixel[0], self.rightDoor_pixel[1])[0]) == 0:
                self.get_logger().warn('[go_to_right_door]: rightDoor_pixel: (%d, %d) is not reachable' % (self.rightDoor_pixel[0], self.rightDoor_pixel[1]))
            else:
                self.get_logger().info('[go_to_right_door]: going to rightDoor_pixel: (%d, %d)' % (self.rightDoor_pixel[0], self.rightDoor_pixel[1]))
                self.move_to(self.rightDoor_pixel[0], self.rightDoor_pixel[1])

        elif self.state == "rotate_to_left_door":
            # this assume that the robot is started straight and door is perpendicular to the robot starting yaw
            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)
                
            # roate to face left door, whichis at yaw = 180
            deltaAngle = Float64()
            deltaAngle.data = 180.0 - self.yaw
            
            # add safe guard of 180
            if deltaAngle.data == 180:
                deltaAngle.data = 179
            
            self.deltaAngle_publisher.publish(deltaAngle)
            self.state = "maze_rotating"
            
            # # temp
            # self.magicState = "idle"
            
            # set magicState to be enter_to_left_door, so that once rotated to face door, it will enter_to_left_door
            self.magicState = "enter_to_left_door"
            
        elif self.state == "rotate_to_right_door":
            # this assume that the robot is started straight and door is perpendicular to the robot starting yaw
            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)
            
            # roate to face left door, whichis at yaw = 0
            deltaAngle = Float64()
            deltaAngle.data = 0.0 - self.yaw
            
            # add safe guard of 180
            if deltaAngle.data == 180:
                deltaAngle.data = 179
            
            self.deltaAngle_publisher.publish(deltaAngle)
            self.state = "maze_rotating"
            
            # # temp
            # self.magicState = "idle"
            
            # set magicState to be enter_to_right_door, so that once rotated to face door, it will enter_to_right_door
            self.magicState = "enter_to_right_door"
            
        elif self.state == "enter_to_left_door":
            # move forward until front is within BUCKET_TOO_CLOSE
            if not (any(self.laser_range[0:self.bucketFrontLeftIndex] < BUCKET_TOO_CLOSE) or any(self.laser_range[self.bucketFrontRightIndex:] < BUCKET_TOO_CLOSE)):
                
                # move forwad
                self.linear_msg.data = LIN_MAX // 2
                self.linear_publisher.publish(self.linear_msg)
                
                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)       
                
            else:
                # move forwad
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)
                
                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)       
                
                # set magicState to start bucket task after moving in to room
                self.state = self.magicState = "checking_walls_distance"
                
                # set boolCurve to 1, to be place in the state before checking_walls_distance
                boolCurve_msg = Int8()
                boolCurve_msg.data = 1
                self.boolCurve_publisher.publish(boolCurve_msg)
                
                # disable OCC callbacks since after entering it will not be used
                self.disableOCC = True
                
                self.get_logger().info('[enter_to_left_door]: setting newOccFlag: %s, disableOCC: %s' % (str(self.newOccFlag), str(self.disableOCC)))
                
                # # temp
                # self.state = self.magicState = "idle"
            
        elif self.state == "enter_to_right_door":
            # move forward until front is within BUCKET_TOO_CLOSE
            if not (any(self.laser_range[0:self.bucketFrontLeftIndex] < BUCKET_TOO_CLOSE) or any(self.laser_range[self.bucketFrontRightIndex:] < BUCKET_TOO_CLOSE)):
                
                # move forwad
                self.linear_msg.data = LIN_MAX // 2
                self.linear_publisher.publish(self.linear_msg)
                
                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)       
            
            else:
                # move forwad
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)
                
                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)   
                
                # set magicState to start bucket task after moving in to room
                self.state = self.magicState = "checking_walls_distance"
                
                # set boolCurve to 1, to be place in the state before checking_walls_distance
                boolCurve_msg = Int8()
                boolCurve_msg.data = 1
                self.boolCurve_publisher.publish(boolCurve_msg)
                
                # disable OCC callbacks since after entering it will not be used
                self.disableOCC = True
                
                self.get_logger().info('[enter_to_right_door]: setting newOccFlag: %s, disableOCC: %s' % (str(self.newOccFlag), str(self.disableOCC)))
                
                # # temp
                # self.state = self.magicState = "idle"

        elif self.state == "checking_walls_distance":
            # lidar minimum is 12 cm send by node, datasheet says 16 cm
            # by experimentation need 30 cm
            # if less than 30 cm from nearest object, move away from it, else can find the bucket using bucketFinderNode
            # bucket finder doesnt work if its too close to wall

            argmin = np.nanargmin(self.laser_range)
            angle_min = self.index_to_angle(argmin, self.range_len)
            self.get_logger().info('[checking_walls_distance]: angle_min %f' % angle_min)

            min_distance = self.laser_range[argmin]

            self.get_logger().info('[checking_walls_distance]: min_distance %f' % min_distance)

            if min_distance < BUCKET_TOO_CLOSE:
                self.get_logger().info('[checking_walls_distance]: too close! moving away')

                # set linear to be zero
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)

                # angle_min > or < 180, the delta angle to move away from the object is still the same
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = angle_min - 180.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)

                self.state = self.magicState = "rotating_to_move_away_from_walls"

            else:
                self.state = self.magicState = "rotating_to_bucket"
        elif self.state == "rotating_to_move_away_from_walls":
            # if still rotating wait, else can move forward until the back is 30 cm away
            if self.robotControlNodeState == "rotateByAngle":
                self.get_logger().info('[rotating_to_move_away_from_walls]: still rotating, waiting')
                pass
            else:
                # get the index of the front left, front right, back, left, right
                # move until the back is more than 40 cm or stop if the front is less than 30 cm
                # 40cm must be more than the 30cm from smallest distance, so that it wont rotate and get diff distance, lidar is not the center of rotation
                # must use any not all incase of NaN

                # if front and butt not clear, rotate to left or right with the most space, then pass back to rotating_to_move_away_from_walls
                # if front is not clear, stop and pass to checking_walls_distance to rotate away from closest object
                # if butt is not clear, keep going forwad until clear then pass to checking_walls_distance to check again
                frontNotClear = any(self.laser_range[0:self.bucketFrontLeftIndex] < BUCKET_TOO_CLOSE) or any(self.laser_range[self.bucketFrontRightIndex:] < BUCKET_TOO_CLOSE) or np.all(np.isnan(self.laser_range[self.bucketFrontRightIndex:])) or np.all(np.isnan(self.laser_range[0:self.bucketFrontLeftIndex]))
                buttNotClear = any(self.laser_range[self.backIndexL:self.backIndexH] < BUCKET_TOO_CLOSE + 0.10) or np.all(np.isnan(self.laser_range[self.backIndexL:self.backIndexH]))
                
                if frontNotClear and buttNotClear:
                    self.get_logger().info('[rotating_to_move_away_from_walls]: front and butt got somthing')
                    
                    if np.nanmean(self.laser_range[self.leftIndexL:self.leftIndexH]) > np.nanmean(self.laser_range[self.rightIndexL:self.rightIndexH]):
                        self.get_logger().info('[rotating_to_move_away_from_walls]: rotate to left')
                        deltaAngle_msg = Float64()
                        deltaAngle_msg.data = 90.0
                        self.deltaAngle_publisher.publish(deltaAngle_msg)
                    else:
                        self.get_logger().info('[rotating_to_move_away_from_walls]: rotate to right')
                        deltaAngle_msg = Float64()
                        deltaAngle_msg.data = -90.0
                        self.deltaAngle_publisher.publish(deltaAngle_msg)
                   
                elif frontNotClear:
                    # infront got something
                    self.get_logger().info('[rotating_to_move_away_from_walls]: something infront')

                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)

                    # set delta angle = 0 to stop
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = 0.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)

                    # send back to checking_walls_distance to check for all distance again
                    self.state = self.magicState = "checking_walls_distance"

                else:
                    self.get_logger().info('[rotating_to_move_away_from_walls]: front is still clear! go forward')

                    if buttNotClear:
                        self.get_logger().info('[rotating_to_move_away_from_walls]: butt is still near! go forward')

                        # set linear to be self.linear_speed to move forward fastest
                        self.linear_msg.data = LIN_MAX // 2
                        self.linear_publisher.publish(self.linear_msg)

                        anglularVel_msg = Int8()

                        # if left got something, rotate right
                        # elif right got something, rotate left
                        # else go straight
                        if all(self.laser_range[self.leftIndexL:self.leftIndexH] < BUCKET_TOO_CLOSE):
                            anglularVel_msg.data = -127
                            self.get_logger().info('[rotating_to_move_away_from_walls]: moving forward and right')
                        elif all(self.laser_range[self.rightIndexL:self.rightIndexH] < BUCKET_TOO_CLOSE):
                            anglularVel_msg.data = 127
                            self.get_logger().info('[rotating_to_move_away_from_walls]: moving forward and left')
                        else:
                            anglularVel_msg.data = 0
                            self.get_logger().info('[rotating_to_move_away_from_walls]: moving forward')

                        self.anglularVel_publisher.publish(anglularVel_msg)
                    else:
                        # moved far enough
                        self.get_logger().info('[rotating_to_move_away_from_walls]: moved far enough, butt is clear')

                        # set linear to be zero
                        self.linear_msg.data = 0
                        self.linear_publisher.publish(self.linear_msg)

                        # set delta angle = 0 to stop
                        deltaAngle_msg = Float64()
                        deltaAngle_msg.data = 0.0
                        self.deltaAngle_publisher.publish(deltaAngle_msg)

                        # send back to checking_walls_distance to check for all distance again
                        self.state = self.magicState = "checking_walls_distance"

        # elif self.state == "rotating_to_bucket":
        #     # if close to forward, go to next state, else align to bucket first
        #     if abs(self.bucketAngle) < 2:
        #         self.get_logger().info('[rotating_to_bucket]: close enough, moving to bucket now')
        #         self.state = self.magicState = "moving_to_bucket"
        #     else:
        #         self.get_logger().info('[rotating_to_bucket]: rotating to face bucket')

        #         if self.bucketAngle < 180:
        #             # set linear to be zero
        #             self.linear_msg.data = 0
        #             self.linear_publisher.publish(self.linear_msg)

        #             # set delta angle = bucketAngle
        #             deltaAngle_msg = Float64()
        #             deltaAngle_msg.data = self.bucketAngle * 1.0  # to change int to float type
        #             self.deltaAngle_publisher.publish(deltaAngle_msg)
        #         elif self.bucketAngle > 180:
        #             # set linear to be zero
        #             self.linear_msg.data = 0
        #             self.linear_publisher.publish(self.linear_msg)

        #             # set delta angle = bucketAngle -360
        #             deltaAngle_msg = Float64()
        #             deltaAngle_msg.data = self.bucketAngle - 360.0  # to change int to float type
        #             self.deltaAngle_publisher.publish(deltaAngle_msg)
        #         else:
        #             # the case where it is 180, turn 90 deg first
        #             # set linear to be zero
        #             self.linear_msg.data = 0
        #             self.linear_publisher.publish(self.linear_msg)

        #             # set delta angle = 90
        #             deltaAngle_msg = Float64()
        #             deltaAngle_msg.data = 90.0
        #             self.deltaAngle_publisher.publish(deltaAngle_msg)

        #         # send to moving_to_bucket to wait for rotation to finish
        #         self.state = self.magicState = "moving_to_bucket"

        #         # on limit switch
        #         switch_msg = String()
        #         switch_msg.data = "activate"
        #         self.switch_publisher.publish(switch_msg)
        # elif self.state == "moving_to_bucket":
        #     # if still rotating wait, else can move forward until hit bucket
        #     if self.robotControlNodeState == "rotateByAngle":
        #         self.get_logger().info('[moving_to_bucket]: still rotating, waiting')
        #         pass
        #     else:
        #         # set linear to be self.linear_speed to move forward fastest
        #         self.linear_msg.data = LIN_MAX
        #         self.linear_publisher.publish(self.linear_msg)

        #         # if the bucket is in the to the right, turn left slightly
        #         anglularVel_msg = Int8()
                
        #         # if infront ish, so angle to change is small, can move forward while turning
        #         # else throw back to checking_walls_distance
        #         # if 5 < self.bucketAngle < 45:
        #         #     anglularVel_msg.data = 100
        #         #     self.get_logger().info('[moving_to_bucket]: moving forward and left')
        #         # elif 315 < self.bucketAngle < 355:
        #         #     anglularVel_msg.data = -100
        #         #     self.get_logger().info('[moving_to_bucket]: moving forward and right')
        #         # elif 45 <= self.bucketAngle <= 315:
        #         #     self.get_logger().info('[moving_to_bucket]: go back checking_walls_distance, self.bucketAngle = %f' % self.bucketAngle)
        #         #     self.state = self.magicState = "checking_walls_distance"    
        #         # else:
        #         #     anglularVel_msg.data = 0
        #         #     self.get_logger().info('[moving_to_bucket]: moving forward')
                
        #         if self.bucketAngle > 5 and self.bucketAngle < 180:
        #             anglularVel_msg.data = 100
        #             self.get_logger().info('[moving_to_bucket]: moving forward and left')
        #         elif self.bucketAngle < 355 and self.bucketAngle > 180:
        #             anglularVel_msg.data = -100
        #             self.get_logger().info('[moving_to_bucket]: moving forward and right')
        #         else:
        #             anglularVel_msg.data = 0
        #             self.get_logger().info('[moving_to_bucket]: moving forward')

        #         self.anglularVel_publisher.publish(anglularVel_msg)

        #         # if the bucket is hit, the state transition and stopping will be done by the switch_listener_callback
        #     pass
        
        # elif self.state == "rotating_to_move_away_from_walls":
        #     # if still rotating wait, else can move forward until the back is 30 cm away
        #     if self.robotControlNodeState == "rotateByAngle":
        #         self.get_logger().info('[rotating_to_move_away_from_walls]: still rotating, waiting')
        #         pass
        #     else:
        #         # get the index of the front left, front right, back, left, right
        #         # move until the back is more than 40 cm or stop if the front is less than 30 cm
        #         # 40cm must be more than the 30cm from smallest distance, so that it wont rotate and get diff distance, lidar is not the center of rotation
        #         # must use any not all incase of NaN
        #         if any(self.laser_range[0:self.bucketFrontLeftIndex] < BUCKET_TOO_CLOSE) or any(
        #                 self.laser_range[self.bucketFrontRightIndex:] < BUCKET_TOO_CLOSE):
        #             # infront got something
        #             self.get_logger().info('[rotating_to_move_away_from_walls]: something infront')

        #             # set linear to be zero
        #             self.linear_msg.data = 0
        #             self.linear_publisher.publish(self.linear_msg)

        #             # set delta angle = 0 to stop
        #             deltaAngle_msg = Float64()
        #             deltaAngle_msg.data = 0.0
        #             self.deltaAngle_publisher.publish(deltaAngle_msg)

        #             # send back to checking_walls_distance to check for all distance again
        #             self.state = self.magicState = "checking_walls_distance"

        #         else:
        #             self.get_logger().info('[rotating_to_move_away_from_walls]: front is still clear! go forward')

        #             if any(self.laser_range[self.backIndexL:self.backIndexH] < BUCKET_TOO_CLOSE + 0.10):
        #                 self.get_logger().info('[rotating_to_move_away_from_walls]: butt is still near! go forward')

        #                 # set linear to be self.linear_speed to move forward fastest
        #                 self.linear_msg.data = LIN_MAX
        #                 self.linear_publisher.publish(self.linear_msg)

        #                 anglularVel_msg = Int8()

        #                 # if left got something, rotate right
        #                 # elif right got something, rotate left
        #                 # else go straight
        #                 if all(self.laser_range[self.leftIndexL:self.leftIndexH] < BUCKET_TOO_CLOSE):
        #                     anglularVel_msg.data = -LIN_MAX
        #                     self.get_logger().info('[rotating_to_move_away_from_walls]: moving forward and right')
        #                 elif all(self.laser_range[self.rightIndexL:self.rightIndexH] < BUCKET_TOO_CLOSE):
        #                     anglularVel_msg.data = LIN_MAX
        #                     self.get_logger().info('[rotating_to_move_away_from_walls]: moving forward and left')
        #                 else:
        #                     anglularVel_msg.data = 0
        #                     self.get_logger().info('[rotating_to_move_away_from_walls]: moving forward')

        #                 self.anglularVel_publisher.publish(anglularVel_msg)
        #             else:
        #                 # moved far enough
        #                 self.get_logger().info('[rotating_to_move_away_from_walls]: moved far enough, butt is clear')

        #                 # set linear to be zero
        #                 self.linear_msg.data = 0
        #                 self.linear_publisher.publish(self.linear_msg)

        #                 # set delta angle = 0 to stop
        #                 deltaAngle_msg = Float64()
        #                 deltaAngle_msg.data = 0.0
        #                 self.deltaAngle_publisher.publish(deltaAngle_msg)

        #                 # send back to checking_walls_distance to check for all distance again
        #                 self.state = self.magicState = "checking_walls_distance"

        elif self.state == "rotating_to_bucket":
            # if close to forward, go to next state, else align to bucket first
            if abs(self.bucketAngle) < 2:
                self.get_logger().info('[rotating_to_bucket]: close enough, moving to bucket now')
                self.state = self.magicState = "moving_to_bucket"
            else:
                self.get_logger().info('[rotating_to_bucket]: rotating to face bucket')

                if self.bucketAngle < 180:
                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)
                    
                    time.sleep(1)

                    # set delta angle = bucketAngle
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = self.bucketAngle * 1.0  # to change int to float type
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                    
                    self.get_logger().info('[rotating_to_bucket]: rotating self.bucketAngle = %f' % self.bucketAngle)
                    
                elif self.bucketAngle > 180:
                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)
                    
                    time.sleep(1)

                    # set delta angle = bucketAngle -360
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = self.bucketAngle - 360.0  # to change int to float type
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                    
                    self.get_logger().info('[rotating_to_bucket]: rotating self.bucketAngle = %f' % self.bucketAngle)
                else:
                    # the case where it is 180, turn 179 deg
                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)
                    
                    time.sleep(1)

                    # set delta angle = 90
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = 179.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)

                # send to moving_to_bucket to wait for rotation to finish
                self.state = self.magicState = "moving_to_bucket"
                
                # start bucket timer
                self.bucketStarted = time.time()
                
                # on limit switch
                switch_msg = String()
                switch_msg.data = "activate"
                self.switch_publisher.publish(switch_msg)
        elif self.state == "moving_to_bucket":
            # if still rotating wait, else can move forward until hit bucket
            if self.robotControlNodeState == "rotateByAngle":
                self.get_logger().info('[moving_to_bucket]: still rotating, waiting')
                pass
            else:
                # if after 30s still not hit, change back to checking_walls_distance
                if time.time() - self.bucketStarted < 30:
                    self.linear_msg.data = 38
                    self.linear_publisher.publish(self.linear_msg)

                    # if the bucket is in the to the right, turn left slightly
                    anglularVel_msg = Int8()
                    
                    # bucket angle becomes unreliable at close range so if the bucket is close enough, just go straight
                    
                    if any(self.laser_range[0:self.bucketFrontLeftIndex] < 0.30) or any(self.laser_range[self.bucketFrontRightIndex:] < 0.30) or np.all(np.isnan(self.laser_range[self.bucketFrontRightIndex:])) or np.all(np.isnan(self.laser_range[0:self.bucketFrontLeftIndex])):
                        anglularVel_msg.data = 0
                        self.get_logger().info('[moving_to_bucket]: too close, just moving forward')
                    else:
                        if self.bucketAngle > 5 and self.bucketAngle < 180:
                            anglularVel_msg.data = 64
                            self.get_logger().info('[moving_to_bucket]: moving forward and left')
                        elif self.bucketAngle < 355 and self.bucketAngle > 180:
                            anglularVel_msg.data = -64
                            self.get_logger().info('[moving_to_bucket]: moving forward and right')
                        else:
                            anglularVel_msg.data = 0
                            self.get_logger().info('[moving_to_bucket]: moving forward')

                    self.anglularVel_publisher.publish(anglularVel_msg)

                    # if the bucket is hit, the state transition and stopping will be done by the switch_listener_callback
                    pass
                else:
                    # off limit switch
                    switch_msg = String()
                    switch_msg.data = "deactivate"
                    self.switch_publisher.publish(switch_msg)
                    
                    # change state back to checking_walls_distance
                    self.state = self.magicState = "checking_walls_distance"
                    
        elif self.state == "releasing":
            servoAngle_msg = UInt8()
            servoAngle_msg.data = 180
            self.servo_publisher.publish(servoAngle_msg)
            self.get_logger().info('[releasing]: easy clap')

            # 5 second after releasing, go back to idle
            if (time.time() - self.lastState) > 5:
                self.state = self.magicState = "idle"

        else:
            self.get_logger().error('state %s not defined' % self.state)

        ''' ================================================ DEBUG PLOT ================================================ '''
        try:
            if self.show_plot and len(self.dilutedOccupancyMap) > 0 and (time.time() - self.lastPlot) > 1:
                PLOT_ORI = False
                PLOT_DILUTED = False
                PLOT_PROCESSED = True
                
                # PLOT_ORI = False
                # PLOT_DILUTED = True
                # PLOT_PROCESSED = False
                
                # PLOT_ORI = True
                # PLOT_DILUTED = False
                # PLOT_PROCESSED = False
                
                # Pixel values
                ROBOT = 0
                # UNMAPPED = 1
                # OPEN = 2
                # OBSTACLE = 3
                MAGIC_ORIGIN = 4
                ESTIMATE_DOOR = 5
                FINISH_LINE = 6
                FRONTIER = 7
                FRONTIER_POINT = 8
                PATH_PLANNING_POINT = 9

                if PLOT_DILUTED == True:
                    # shows the diluted occupancy map with frontiers and path planning points
                    self.totalMap = self.dilutedOccupancyMap.copy()

                    # add padding until certain size, add in the estimated door and finish line incase they exceed for whatever reason
                    TARGET_SIZE_M = 5
                    TARGET_SIZE_p = max(round(TARGET_SIZE_M / self.map_res), self.leftDoor_pixel[1], self.leftDoor_pixel[0], self.rightDoor_pixel[1], self.rightDoor_pixel[0], self.finishLine_pixel[1], self.finishLine_pixel[0])

                    # Calculate the necessary padding
                    padding_height = max(0, TARGET_SIZE_p - self.totalMap.shape[0])
                    padding_width = max(0, TARGET_SIZE_p - self.totalMap.shape[1])

                    # Define the number of pixels to add to the height and width
                    padding_height = (0, padding_height)  # Replace with the number of pixels you want to add to the top and bottom
                    padding_width = (0, padding_width)  # Replace with the number of pixels you want to add to the left and right

                    # Pad the image
                    self.totalMap = np.pad(self.totalMap, pad_width=(padding_height, padding_width), mode='constant', constant_values=UNMAPPED)

                    try:
                        # Set the value of the door esitmate and finish line, y and x are flipped becasue image coordinates are (row, column)
                        self.totalMap[self.leftDoor_pixel[1], self.leftDoor_pixel[0]] = ESTIMATE_DOOR
                        self.totalMap[self.rightDoor_pixel[1], self.rightDoor_pixel[0]] = ESTIMATE_DOOR

                        self.totalMap[self.finishLine_pixel[1], self.finishLine_pixel[0]] = FINISH_LINE
                    except:
                        self.get_logger().info('[Debug Plotter]: door and finish line cannot plot')

                    # Set the value of the frontier and the frontier points
                    for pixel in self.frontier:
                        self.totalMap[pixel[0], pixel[1]] = FRONTIER

                    for pixel in self.frontierPoints:
                        self.totalMap[pixel[1], pixel[0]] = FRONTIER_POINT

                    # Set the value for the path planning points
                    for i in range(len(self.dest_x)):
                        self.totalMap[self.dest_y[i]][self.dest_x[i]] = PATH_PLANNING_POINT

                    colourList = ['black',
                                (85/255, 85/255, 85/255),         # dark grey
                                (170/255, 170/255, 170/255),      # light grey
                                'white',
                                (50/255, 205/255, 50/255),        # lime green
                                (1, 1, 0),                        # yellow
                                (0, 1, 0)                         # green
                                ]

                    # add in colours for each type of pixel
                    if len(self.frontier) > 0:
                        colourList.append((0, 1, 1))    # cyan

                    if len(self.frontierPoints) > 0:
                        colourList.append((1, 0, 1))    # magenta

                    if len(self.dest_x) > 0:
                        colourList.append((1, 165/255, 0))   # orange

                    # set bot pixel to 0, y and x are flipped becasue image coordinates are (row, column)
                    self.totalMap[self.boty_pixel][self.botx_pixel] = ROBOT

                    # set magic origin pixel to 7, y and x are flipped becasue image coordinates are (row, column)
                    self.totalMap[self.magicOriginy_pixel][self.magicOriginx_pixel] = MAGIC_ORIGIN

                    # MAGIC_ORIGIN will override ROBOT and colour will be weird, if robot at magic origin

                    cmap = ListedColormap(colourList)

                    plt.imshow(self.totalMap, origin='lower', cmap=cmap)

                    plt.draw_all()
                    # pause to make sure the plot gets created
                    plt.pause(0.00000000001)
                elif PLOT_PROCESSED == True:
                    self.totalMap = self.processedOcc.copy()
                    
                    # Normalize the array to the range 0-255
                    totalMap_normalized = ((self.totalMap - 0) * (255 - 0) / (100 - 0)).astype(np.uint8)

                    # Convert the normalized array to integers
                    totalMap_int = totalMap_normalized.astype(np.uint8)

                    # add padding until certain size, add in the estimated door and finish line incase they exceed for whatever reason
                    TARGET_SIZE_M = 5
                    
                    # find the max x and y in must visit
                    # need to check that its not empty
                    if len(self.mustVisitPointsChecked_pixel) > 0:
                        # Find the maximum x and y values
                        maxMustVisitPointsChecked_pixel_x = max(self.mustVisitPointsChecked_pixel, key=lambda point: point[0])[0]
                        maxMustVisitPointsChecked_pixel_y = max(self.mustVisitPointsChecked_pixel, key=lambda point: point[1])[1]
                    else:
                        # set to 0
                        maxMustVisitPointsChecked_pixel_x = 0
                        maxMustVisitPointsChecked_pixel_y = 0

                    # must add 10 otherwise cant plot the far point
                    TARGET_SIZE_p = max(round(TARGET_SIZE_M / self.map_res), self.leftDoor_pixel[1], self.leftDoor_pixel[0], self.rightDoor_pixel[1], self.rightDoor_pixel[0], self.finishLine_pixel[1], self.finishLine_pixel[0], maxMustVisitPointsChecked_pixel_x, maxMustVisitPointsChecked_pixel_y) + 10

                    # Calculate the necessary padding
                    padding_height = max(0, TARGET_SIZE_p - self.totalMap.shape[0])
                    padding_width = max(0, TARGET_SIZE_p - self.totalMap.shape[1])

                    # Define the number of pixels to add to the height and width
                    padding_height = (0, padding_height)  # Replace with the number of pixels you want to add to the top and bottom
                    padding_width = (0, padding_width)  # Replace with the number of pixels you want to add to the left and right

                    # Pad the image
                    totalMap_rgb = np.pad(totalMap_int, pad_width=(padding_height, padding_width), mode='constant', constant_values=0)

                    # Convert the single-channel grayscale image to a three-channel RGB image
                    totalMap_rgb = cv2.cvtColor(totalMap_rgb, cv2.COLOR_GRAY2BGR)
                    
                    try:
                        # Set the value of the door esitmate and finish line, y and x are flipped becasue image coordinates are (row, column)
                        totalMap_rgb[self.leftDoor_pixel[1]][self.leftDoor_pixel[0]] = (50, 205, 50)
                        totalMap_rgb[self.rightDoor_pixel[1]][self.rightDoor_pixel[0]] = (50, 205, 50)
                        totalMap_rgb[self.finishLine_pixel[1]][self.finishLine_pixel[0]] = (50, 205, 50)
                        
                        # plot the must visit points
                        for x, y in self.mustVisitPointsChecked_pixel:
                            totalMap_rgb[y][x] = (50, 205, 50)
                    except:
                        self.get_logger().info('[Debug Plotter]: door and finish line cannot plot')
                        
                    # Set the value for the path planning points
                    for i in range(len(self.dest_x)):
                        totalMap_rgb[self.dest_y[i]][self.dest_x[i]] = (255, 165, 0)
                        
                    # Set the value of the frontier and the frontier points
                    for pixel in self.frontier:
                        totalMap_rgb[pixel[0]][pixel[1]] = (0, 255, 255)

                    for pixel in self.frontierPoints:
                        totalMap_rgb[pixel[1]][pixel[0]] = (255, 0, 255)
                        
                    # set bot pixel to 0, y and x are flipped becasue image coordinates are (row, column)
                    totalMap_rgb[self.boty_pixel][self.botx_pixel] = (255, 0, 0)

                    # set magic origin pixel to 7, y and x are flipped becasue image coordinates are (row, column)
                    totalMap_rgb[self.magicOriginy_pixel][self.magicOriginx_pixel] = (255, 255, 0)

                    # Display the image using matplotlib.pyplot
                    plt.imshow(cv2.cvtColor(totalMap_rgb, cv2.COLOR_BGR2RGB), origin='lower')
                    plt.draw_all()
                    plt.pause(0.00000000001)
                    
                elif PLOT_ORI == True:
                    # plt.imshow(self.occupancyMap, origin='lower')

                    # cmap = ListedColormap(['black', 'red', 'gray'])
                    # plt.imshow(self.processedOcc, cmap='gray', origin='lower')
                    plt.imshow(self.oriorimap, cmap='gray', origin='lower')

                    plt.draw_all()
                    # pause to make sure the plot gets created
                    plt.pause(0.00000000001)

                self.lastPlot = time.time()
        except:
            self.get_logger().info('[Debug Plotter]: Debug cannot plot')

    def move_straight_to(self, tx, ty):
        target_yaw = math.atan2(ty - self.boty_pixel, tx - self.botx_pixel) * (180 / math.pi)
        self.get_logger().info('[move_straight_to]: currently at (%d %d) with yaw %f, moving straight to (%d, %d)' % (
        self.botx_pixel, self.boty_pixel, self.yaw, tx, ty))
        # self.get_logger().info('currently yaw is %f, target yaw is %f' % (self.yaw, target_yaw))
        
        # if deltaAngle is too big, stop then rotate
        # else, rotate and move at the same time
        if abs(target_yaw - self.yaw) >= 45:
            # set linear
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)
        else:
            # set linear
            self.linear_msg.data = LIN_WHEN_ROTATING
            self.linear_publisher.publish(self.linear_msg)
            
        # wait abit first
        time.sleep(0.5)

        # set delta angle to rotate to target angle
        deltaAngle = Float64()
        deltaAngle.data = target_yaw - self.yaw
        self.deltaAngle_publisher.publish(deltaAngle)
        self.state = "maze_rotating"
        
    def toId(self, y, x, d):
        return d * self.map_h * self.map_w + y * self.map_w + x

    def construct_graph(self):
        if self.d_dim == (self.map_h, self.map_w):
            iter = 0
            for y in range(self.map_h):
                for x in range(self.map_w):
                    for d in range(len(self.dx)):
                        for i in [1, -1]:
                            self.d_data[iter] = 3
                            iter += 1
                        ny = y + self.dy[d]
                        nx = x + self.dx[d]
                        if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
                            self.d_data[iter] = self.d_cost[self.processedOcc[ny][nx]]
                            iter += 1
        else:
            self.get_logger().info("[construct_graph]: dimension changed")
            self.d_dim = (self.map_h, self.map_w)
            row = []
            col = []
            data = []
            for y in range(self.map_h):
                for x in range(self.map_w):
                    for d in range(len(self.dx)):
                        for i in [1, -1]:
                            row.append(self.toId(y, x, d))
                            col.append(self.toId(y, x, (d + i) % 4))
                            data.append(3)
                        ny = y + self.dy[d]
                        nx = x + self.dx[d]
                        if 0 <= ny < self.map_h and 0 <= nx < self.map_w:
                            row.append(self.toId(y, x, d))
                            col.append(self.toId(ny, nx, d))
                            data.append(self.d_cost[self.processedOcc[ny][nx]])
            self.d_row = np.array(row)
            self.d_col = np.array(col)
            self.d_data = np.array(data, dtype=np.float32)

    def dijkstra(self):
        dickStarTime = time.time()

        sx = self.botx_pixel
        sy = self.boty_pixel
        cur_dir = round(self.yaw / 90) % 4

        self.construct_graph()

        timeTaken = time.time() - dickStarTime
        self.get_logger().info('[dijkstra1]: it took: %f' % timeTaken)

        graph_size = self.map_h * self.map_w * len(self.dx)
        graph = csr_matrix((self.d_data, (self.d_row, self.d_col)), shape=(graph_size, graph_size))

        timeTaken = time.time() - dickStarTime
        self.get_logger().info('[dijkstra2]: it took: %f' % timeTaken)

        p_dist, p_pre = dijkstra(graph, indices=self.toId(sy, sx, cur_dir), return_predecessors=True)

        timeTaken = time.time() - dickStarTime
        self.get_logger().info('[dijkstra3]: it took: %f' % timeTaken)

        self.dist = np.full((self.map_h, self.map_w), np.inf, dtype=float)
        self.pre = np.full((self.map_h, self.map_w, 2), -1)
        for y in range(self.map_h):
            for x in range(self.map_w):
                mn = np.inf
                opt_d = -1
                for d in range(len(self.dx)):
                    if p_dist[self.toId(y, x, d)] < mn:
                        mn = p_dist[self.toId(y, x, d)]
                        opt_d = d
                if opt_d == -1:
                    continue
                self.dist[y][x] = mn
                p = p_pre[self.toId(y, x, opt_d)]
                if p >= 0:
                    self.pre[y][x] = (p // self.map_w % self.map_h, p % self.map_w)

        timeTaken = time.time() - dickStarTime
        self.get_logger().info('[dijkstra]: it took: %f' % timeTaken)

    def find_path_to(self, tx, ty):
        sx = self.botx_pixel
        sy = self.boty_pixel

        if self.dist[ty][tx] == np.inf:
            self.get_logger().info('[path_finding]: no path from cell (%d %d) to cell (%d %d)' % (sx, sy, tx, ty))
            return [], []

        self.get_logger().info(
            '[path_finding]: distance from cell (%d %d) to cell (%d %d) is %f' % (sx, sy, tx, ty, self.dist[ty][tx]))

        res_x = []
        res_y = []
        while True:
            if len(res_x) >= 2:
                if (res_x[-2] - tx) * (res_y[-1] - ty) == (res_x[-1] - tx) * (res_y[-2] - ty):
                    res_x.pop()
                    res_y.pop()
            # if len(res_x) >= 2:
            #     if abs(res_x[-1] - tx) + abs(res_y[-1] - ty) <= RADIUS_OF_IGNORE:
            #         res_x.pop()
            #         res_y.pop()
            res_x.append(tx)
            res_y.append(ty)
            if ty == sy and tx == sx:
                break
            ty, tx = self.pre[ty][tx]
        res_x.reverse()
        res_y.reverse()
        if len(res_x) >= 2:
            if abs(res_x[0] - res_x[1]) + abs(res_y[0] - res_y[1]) <= RADIUS_OF_IGNORE:
                res_x.pop(1)
                res_y.pop(1)

        self.get_logger().info('[path_finding]: x: %s, y: %s' % (str(res_x), str(res_y)))

        return res_x, res_y

    def move_to(self, tx, ty):
        self.get_logger().info(
            '[move_to]: currently at (%d %d), moving to (%d, %d)' % (self.botx_pixel, self.boty_pixel, tx, ty))
        self.dest_x, self.dest_y = self.find_path_to(tx, ty)

        if len(self.dest_x) == 0:
            self.get_logger().info('[move_to]: no path found get back to magicState: %s' % self.magicState)
            self.state = self.magicState
        else:
            self.state = "maze_moving"

    def frontierSearch(self):
        if len(self.dilutedOccupancyMap) == 0:
            return

        # 0 = robot
        # 1 = unmapped
        # 2 = mapped and open
        # 3 = mapped and obstacle
        # 4 = frontier
        # 5 = frontier point

        ''' ================================================ Frontier Search ================================================ '''
        # frontier is between 1 = unmapped and 2 = mapped and open

        self.frontier = []

        # Iterate over the array
        for i in range(self.map_h):
            for j in range(self.map_w):
                # Check if the current pixel is 2
                if self.dilutedOccupancyMap[i, j] == 2:
                    # check for diagonals also so BFS with UP, DOWN, LEFT, RIGHT can colect all frontier pixels
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            # Skip the current pixel
                            if di == 0 and dj == 0:
                                continue
                            # Check if the neighboring pixel is inside the image
                            if 0 <= i + di < self.map_h and 0 <= j + dj < self.map_w:
                                # Check if the neighboring pixel is 1
                                if self.dilutedOccupancyMap[i + di, j + dj] == 1:
                                    self.frontier.append((i, j))
                                    # self.get_logger().info(str("Pixel 1 at (" + str(i) + ", " + str(j) + ") is next to pixel 2 at (" + str(i + di) + ", " + str(j + dj) + ")" ))

        # check if frontier is empty
        if len(self.frontier) == 0:
            self.frontierPoints = []
        else:
            # BFS to find all frontier groups
            # Initialize the queue with the first pixel
            queue = deque([self.frontier[0]])

            # Initialize the set of visited pixels
            visited = set([self.frontier[0]])

            # Initialize the list of groups
            groups = []

            # Perform the BFS
            while queue:
                # Start a new group
                group = []

                # Process all pixels in the current group
                while queue:
                    i, j = queue.popleft()
                    group.append((i, j))

                    # Check the neighboring pixels
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            # Skip the current pixel and diagonal pixels
                            if (di == 0 and dj == 0) or (di != 0 and dj != 0):
                                continue

                            # Check if the neighboring pixel is inside the image and in the frontier
                            if 0 <= i + di < self.map_h and 0 <= j + dj < self.map_w and (
                            i + di, j + dj) in self.frontier:
                                # Check if the neighboring pixel has not been visited yet
                                if (i + di, j + dj) not in visited:
                                    # Add the neighboring pixel to the queue and the set of visited pixels
                                    queue.append((i + di, j + dj))
                                    visited.add((i + di, j + dj))

                # Add the group to the list of groups
                groups.append(group)

                # Find the next unvisited pixel in the frontier
                for pixel in self.frontier:
                    if pixel not in visited:
                        queue.append(pixel)
                        visited.add(pixel)
                        break

            # find frontier points if the frontier group has more than FRONTIER_THRESHOLD points
            # Initialize the list of frontier points
            self.frontierPoints = []

            # Iterate over the groups
            for group in groups:
                if len(group) < FRONTIER_THRESHOLD:
                    continue

                # Extract the x and y coordinates
                x_coords = [w for h, w in group]
                y_coords = [h for h, w in group]

                # Calculate the middle x and y coordinates
                middle_x = sorted(x_coords)[len(x_coords) // 2]
                middle_y = sorted(y_coords)[len(y_coords) // 2]

                # skip if it is not reachable
                if self.dist[middle_y][middle_x] >= self.frontierSkipThreshold:
                    self.get_logger().info('[frontierSearch]: frontier point (%d %d) is too far at %f; skipped' % (middle_x, middle_y, self.dist[middle_y][middle_x]))
                    continue
                
                # skip if frontier is beyond maze
                if middle_y > self.mazeTopBoundary or middle_y < self.mazeBotBoundary or middle_x < self.mazeLeftBoundary or middle_x > self.mazeRightBoundary:
                    self.get_logger().info('[frontierSearch]: frontier point (%d %d) is beyond maze; skipped' % (middle_x, middle_y))
                    continue

                self.frontierPoints.append((middle_x, middle_y))
                
            # new frontier found after doing must visit points
            if self.magicState == "visiting_must_visit" and len(self.frontierPoints) > 0:
                self.state = self.magicState = "frontier_search"
                return
                
            if self.magicState == "visiting_must_visit":
                # skip must visit points if its beyond MUST_VISIT_COST, other wise add to frontier points to be considered for path planning
                for point in self.mustVisitPointsChecked_pixel:
                    # check if must visit poiints are in the map
                    if 0 < point[0] < self.map_w and 0 < point[1] < self.map_h:
                        
                        # this is to avoid point thick walls
                        if self.dist[point[1]][point[0]] >= MUST_VISIT_COST:
                            self.get_logger().info('[frontierSearch]: must visit point (%d %d) is too far at %f; skipped' % (point[0], point[1], self.dist[point[1]][point[0]]))
                        else:
                            self.frontierPoints.append(point)
                            
                # if using must visit points, sort by dist, so that it will follow the least resistance path
                self.frontierPoints = sorted(self.frontierPoints, key=lambda point: self.dist[point[1]][point[0]])
                return
                
            # Current position
            curr_pos = np.array([self.botx_pixel, self.boty_pixel])
            
            # if the cost is low, means not across wall 
            
            # if froniter is a certain radius from the robot 
            #   filter away those will high cost (if next to robot and still high cost means crossing wall)
            #   then sort by distance from current position, 
            #   this will explore the points around the robot
            # else sort by x/y value base on maze
            
            if any(np.linalg.norm(curr_pos - np.array(point)) < VISIT_RADIUS_M / self.map_res for point in self.frontierPoints):
                # # filter away those will high cost
                # self.frontierPoints = [point for point in self.frontierPoints if self.dist[point[1]][point[0]] < VISIT_COST]
                
                # sort by distance away from current position
                self.get_logger().info('[frontierSearch]: sort by distance away from curr')
                
                def cmp_points(a, b):
                    d_to_a = np.linalg.norm(curr_pos - np.array(a))
                    d_to_b = np.linalg.norm(curr_pos - np.array(b))
                    if d_to_a == d_to_b:
                        return 0
                    return -1 if d_to_a < d_to_b else 1

                self.frontierPoints.sort(key=cmp_to_key(cmp_points))
                
                if len(self.frontierPoints) >= 2:
                    # if the first two points distance are closer than FRONTIER_DIST_M, sort by lower y value first
                    d0 = np.linalg.norm(curr_pos - np.array(self.frontierPoints[0]))
                    d1 = np.linalg.norm(curr_pos - np.array(self.frontierPoints[1]))
                    
                    if abs(d0 - d1) < FRONTIER_DIST_M / self.map_res:
                        # comapre y values
                        if self.frontierPoints[0][1] > self.frontierPoints[1][1]:
                            self.frontierPoints[0], self.frontierPoints[1] = self.frontierPoints[1], self.frontierPoints[0]
                            
                    # to hope to prevent oscillatory behavior
                    self.get_logger().info('[frontierSearch]: distance closest two points')
                             
            else:
                '''
                _________________________________________________
                |                                               |
                |   TOP LEFT        TOP MID          TOP RIGHT  |
                |                                               |
                |                                               |
                |                                               |
                |                                               |
                |                    MAZE                       |
                |                                               |
                |                                               |
                |                                               |
                |______________                                 |
                |  Ori                                          |
                |_______________________________________________|
                
                if exit at TOP RIGHT:   LAST_AT_TOP_RIGHT       decending sort by distance from TOP RIGHT
                if exit at TOP MID:     LAST_AT_TOP_MID         acending sort by distance from TOP MID
                if exit at TOP LEFT:    LAST_AT_TOP_LEFT        decending sort by distance from TOP LEFT
                
                '''
                
                LAST_AT_TOP_LEFT = False
                LAST_AT_TOP_MID = True
                LAST_AT_TOP_RIGHT = False
                
                if LAST_AT_TOP_RIGHT:
                    self.get_logger().info('[frontierSearch]: LAST_AT_TOP_RIGHT')
                    
                    self.frontierPoints = sorted(self.frontierPoints, key=lambda point: point[1]**2 + point[0]**2)
                    
                elif LAST_AT_TOP_LEFT:
                    self.get_logger().info('[frontierSearch]: LAST_AT_TOP_LEFT')
                    
                    self.frontierPoints = sorted(self.frontierPoints, key=lambda point: (point[0] - 3.5 / self.map_res)**2 + point[1]**2)
                    
                elif LAST_AT_TOP_MID:
                    self.get_logger().info('[frontierSearch]: LAST_AT_TOP_MID')
                    
                    self.frontierPoints = list(reversed(sorted(self.frontierPoints, key=lambda point: (point[0] - ((3.5 / 2) / self.map_res))**2 + (point[1] - (2.1 / self.map_res))**2 - 1000000 if (point[1] > (2.1 / self.map_res)) else (point[0] - ((3.5 / 2) / self.map_res))**2 + (point[1] - (2.1 / self.map_res))**2)))
                
            # # sort points by distance from current position
            # # Current position
            # curr_pos = np.array([self.botx_pixel, self.boty_pixel])

            # def cmp_points(a, b):
            #     d_to_a = np.linalg.norm(curr_pos - np.array(a))
            #     d_to_b = np.linalg.norm(curr_pos - np.array(b))
            #     if d_to_a == d_to_b:
            #         return 0
            #     return -1 if d_to_a < d_to_b else 1

            # self.frontierPoints.sort(key=cmp_to_key(cmp_points))
            
            # if len(self.frontierPoints) >= 2:
            #     # if the first two points distance are closer than FRONTIER_DIST_M, sort by lower y value first
            #     d0 = np.linalg.norm(curr_pos - np.array(self.frontierPoints[0]))
            #     d1 = np.linalg.norm(curr_pos - np.array(self.frontierPoints[1]))
                
            #     if abs(d0 - d1) < FRONTIER_DIST_M / self.map_res:
            #         # comapre y values
            #         if self.frontierPoints[0][1] > self.frontierPoints[1][1]:
            #             self.frontierPoints[0], self.frontierPoints[1] = self.frontierPoints[1], self.frontierPoints[0]
                        
            #     # to hope to prevent oscillatory behavior
            #     self.get_logger().info('[frontierSearch]: distance closest two points')
                
            
        self.get_logger().info('[frontierSearch]: frontier points: %s' % str(self.frontierPoints))


def main(args=None):
    rclpy.init(args=args)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start the masterNode.')
    parser.add_argument('-s', type=str, default='n', help='Show plot (y/n)')
    args = parser.parse_args()

    master_node = MasterNode(args.s)

    if args.s == 'y':
        # create matplotlib figure
        plt.ion()
        plt.figure()
    try:
        rclpy.spin(master_node)
    except KeyboardInterrupt:
        pass
    finally:
        master_node.custom_destroy_node()


if __name__ == '__main__':
    main()
