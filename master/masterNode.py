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

occ_bins = [-1, 0, 55, 100]

# CLEARANCE_RADIUS is in cm, used to dilate the obstacles
# radius of turtle bot is around 11 cm
CLEARANCE_RADIUS = 10

# this is in pixel
# FRONTIER_THRESHOLD = 4
FRONTIER_THRESHOLD = 2
# PIXEL_DEST_THRES = 2
PIXEL_DEST_THRES = 2

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

MAZE_FRONT_RANGE = 25
MAZE_FRONT_LEFT_ANGLE = 0 + MAZE_FRONT_RANGE
MAZE_FRONT_RIGHT_ANGLE = 360 - MAZE_FRONT_RANGE

MAZE_CLEARANCE_ANGLE = 10
MAZE_ROTATE_SPEED = 64

# left, right door and finish line coords in meters from the magic origin
LEFT_DOOR_COORDS_M = (1.20, 2.70)
RIGHT_DOOR_COORDS_M = (1.90, 2.70)
FINISH_LINE_M = ((LEFT_DOOR_COORDS_M[0] + RIGHT_DOOR_COORDS_M[0])/2, 2.10)

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
FRONTIER_SKIP_THRESHOLD = 1e9

# distance to two points to be considered sorted by lower y, in meter
FRONTIER_DIST_M = 0.10

# time in s to wait after no more frontier before goin to hall way
WAIT_FRONTIER = 10

# speedssss
LIN_MAX = 120
LIN_WHEN_ROTATING = 120

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
        self.pos_subscription = self.create_subscription(
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

        fsm_period = 0.1  # seconds
        self.fsmTimer = self.create_timer(fsm_period, self.masterFSM)

        self.closestAngle = 0

        # Create a subscriber to the topic fsmDebug
        # to inject state changes for debugging in RQT
        self.pos_subscription = self.create_subscription(
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

        if self.disableOCC == False:
            self.get_logger().info('[occ_callback]: occ enabled')
            
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
                
                if self.magicState == "frontier_search" and self.dist[self.dest_y[-1] + self.offset_y][self.dest_x[-1] + self.offset_x] > FRONTIER_SKIP_THRESHOLD:
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

        if self.magicState == "frontier_search" and self.frontierPoints == []:
            self.get_logger().info('[masterFSM]: no more frontier points go to wait_for_frontier')
            self.state = self.magicState = "wait_for_frontier"
            self.lastWaitForFrontier = time.time()
            
            # set to stop since it sometimes hit to wall during state transistion, as there is not obstacle avoidance outside of moving
            # set linear to be zero
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)

            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)
                
        listStateIgnoreForObstacle = ["idle", 
                                      "checking_walls_distance",
                                      "rotating_to_move_away_from_walls"
                                      "rotating_to_bucket",
                                      "moving_to_bucket",
                                      "releasing",
                                      ]
                
        # special case for maze_rotating
        # if moving then need to check for obstacle, else rotate on spot no need to check
        if (self.state not in listStateIgnoreForObstacle) or (self.state == "maze_rotating" and self.linear_msg.data != 0):
            
            # if obstacle in front and close to both sides, rotate to move between the two
            if any(self.laser_range[:self.mazeFrontLeftindex] < NAV_TOO_CLOSE) \
                    or any(self.laser_range[self.mazeFrontRightindex:] < NAV_TOO_CLOSE) \
                    or np.all(np.isnan(self.laser_range[:self.mazeFrontLeftindex])) \
                    or np.all(np.isnan(self.laser_range[self.mazeFrontRightindex:])):
                self.get_logger().warn('[masterFSM]: ahhh wall to close to front uwu')

                # set linear to be zero
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)

                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)
                
                # move backward for 0.5 sec as long as butt has space
                self.linear_msg.data = -LIN_MAX
                self.linear_publisher.publish(self.linear_msg)
                start = time.time()
                while (time.time() - start < 0.3) \
                    and not any(self.laser_range[self.backIndexL:self.backIndexH] < NAV_TOO_CLOSE) \
                    and not np.all(np.isnan(self.laser_range[self.backIndexL:self.backIndexH])):
                    pass
                
                self.linear_msg.data = 0
                self.linear_publisher.publish(self.linear_msg)
                
                anglularVel_msg = Int8()
                
                # rotate to next point if have next point else just rotate to away from the close one
                if len(self.dest_y) > 1:
                    target_yaw = math.atan2(self.dest_y[0] - self.boty_pixel, self.dest_x[0] - self.botx_pixel) * (
                            180 / math.pi)

                    deltaAngle = target_yaw - self.yaw

                    self.get_logger().info('[masterFSM]: align to next dest: %f' % deltaAngle)

                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)

                    # set delta angle to rotate to target angle
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = deltaAngle * 1.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)

                    self.state = "maze_rotating"
                
                else:
                    self.get_logger().info('[masterFSM]: rotate away from close one')
                    
                    # rotate to away from the close one for 0.5 sec
                    if np.nanmean(self.laser_range[:self.mazeFrontLeftindex]) < np.nanmean(self.laser_range[self.mazeFrontRightindex:]):
                        anglularVel_msg.data = -120
                    else:
                        anglularVel_msg.data = 120
                        
                    self.anglularVel_publisher.publish(anglularVel_msg)
                    
                    start = time.time()
                    while (time.time() - start < 0.75):
                        pass
                
                    anglularVel_msg.data = 0
                    self.anglularVel_publisher.publish(anglularVel_msg)
                
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

            # # set boolCurve to 0
            # boolCurve_msg = Int8()
            # boolCurve_msg.data = 0
            # self.boolCurve_publisher.publish(boolCurve_msg)
            
            # set boolCurve to 1, testing
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
                elif abs(deltaAngle) >= 90:
                    # set linear
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)

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

        elif self.state == "frontier_search":
            # # compare two frontier points and judge which we go first
            # # return True if p1 has higher priority than p2
            # def cmp(p1, p2):
            #     return p1[0] < p2[0]

            # destination = self.frontierPoints[0]
            # for i in range(1, len(self.frontierPoints)):
            #     if cmp(self.frontierPoints[i], destination):
            #         destination = self.frontierPoints[i]

            # Find the point in self.frontierPoints that is closest to the current position
            destination = self.frontierPoints[0]

            self.get_logger().info('[frontier_search]: next destination: (%d, %d)' % (destination[0], destination[1]))

            self.move_to(destination[0], destination[1])
            
        elif self.state == "wait_for_frontier":
            # after WAIT_FRONTIER sec, go to move_to_hallway
            
            self.get_logger().info('[wait_for_frontier]: waitng until %f, now is %f' % (WAIT_FRONTIER, time.time() - self.lastWaitForFrontier))
            
            if len(self.frontierPoints) > 0:
                self.get_logger().info('[wait_for_frontier]: frontier points found go to frontier_search')
                self.state = self.magicState = "frontier_search"
                return
            
            if time.time() - self.lastWaitForFrontier > WAIT_FRONTIER:
                self.get_logger().info('[wait_for_frontier]: time is up go to move_to_hallway')
                self.state = self.magicState = "move_to_hallway"
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
                
            # check if hallway is even in map, if its not, means frontier has not been fully explored, go back to frontier search
            # froniter search changes as confidence in map changes hence sometime when there is no more frontier, but after some time, there is more
            if self.finishLine_pixel[0] < 0 or self.finishLine_pixel[1] < 0 or self.finishLine_pixel[0] >= self.map_w or self.finishLine_pixel[1] >= self.map_h:
                self.get_logger().warn('[move_to_hallway]: finishLine_pixel: (%d, %d) is not reachable is not in map' % (self.finishLine_pixel[0], self.finishLine_pixel[1]))
                
                self.state = self.magicState = "frontier_search"
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
                # if nav works
                self.state = self.magicState = "go_to_left_door"
                
                # if nav is still not working, put bot at centre of both door facing right door and start http_request
                # self.state = self.magicState = "rotate_to_left_door"

            elif self.doorStatus == "door2":
                self.get_logger().info('[http_request]: door2 opened')
                # if nav works
                self.state = self.magicState = "go_to_right_door"
                
                # if nav is still not working, put bot at centre of both door facing right door and start http_request
                # self.state = self.magicState = "rotate_to_right_door"

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
                self.linear_msg.data = LIN_MAX
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
                self.linear_msg.data = LIN_MAX
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
                if any(self.laser_range[0:self.bucketFrontLeftIndex] < BUCKET_TOO_CLOSE) or any(
                        self.laser_range[self.bucketFrontRightIndex:] < BUCKET_TOO_CLOSE):
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

                    if any(self.laser_range[self.backIndexL:self.backIndexH] < BUCKET_TOO_CLOSE + 0.10):
                        self.get_logger().info('[rotating_to_move_away_from_walls]: butt is still near! go forward')

                        # set linear to be self.linear_speed to move forward fastest
                        self.linear_msg.data = LIN_MAX
                        self.linear_publisher.publish(self.linear_msg)

                        anglularVel_msg = Int8()

                        # if left got something, rotate right
                        # elif right got something, rotate left
                        # else go straight
                        if all(self.laser_range[self.leftIndexL:self.leftIndexH] < BUCKET_TOO_CLOSE):
                            anglularVel_msg.data = -LIN_MAX
                            self.get_logger().info('[rotating_to_move_away_from_walls]: moving forward and right')
                        elif all(self.laser_range[self.rightIndexL:self.rightIndexH] < BUCKET_TOO_CLOSE):
                            anglularVel_msg.data = LIN_MAX
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

                    # set delta angle = bucketAngle
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = self.bucketAngle * 1.0  # to change int to float type
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                elif self.bucketAngle > 180:
                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)

                    # set delta angle = bucketAngle -360
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = self.bucketAngle - 360.0  # to change int to float type
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                else:
                    # the case where it is 180, turn 90 deg first
                    # set linear to be zero
                    self.linear_msg.data = 0
                    self.linear_publisher.publish(self.linear_msg)

                    # set delta angle = 90
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = 90.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)

                # send to moving_to_bucket to wait for rotation to finish
                self.state = self.magicState = "moving_to_bucket"

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
                # set linear to be self.linear_speed to move forward fastest
                self.linear_msg.data = LIN_MAX
                self.linear_publisher.publish(self.linear_msg)

                # if the bucket is in the to the right, turn left slightly

                if self.bucketAngle > 5 and self.bucketAngle < 180:
                    anglularVel_msg.data = 100
                    self.get_logger().info('[moving_to_bucket]: moving forward and left')
                elif self.bucketAngle < 355 and self.bucketAngle > 180:
                    anglularVel_msg.data = -100
                    self.get_logger().info('[moving_to_bucket]: moving forward and right')
                else:
                    anglularVel_msg.data = 0
                    self.get_logger().info('[moving_to_bucket]: moving forward')

                self.anglularVel_publisher.publish(anglularVel_msg)

                # if the bucket is hit, the state transition and stopping will be done by the switch_listener_callback
            pass

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
                    
                    # must add 10 otherwise cant plot the far point
                    TARGET_SIZE_p = max(round(TARGET_SIZE_M / self.map_res), self.leftDoor_pixel[1], self.leftDoor_pixel[0], self.rightDoor_pixel[1], self.rightDoor_pixel[0], self.finishLine_pixel[1], self.finishLine_pixel[0]) + 10

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
                    plt.imshow(self.oriorimap, cmap=cmap, origin='lower')

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
        if abs(target_yaw - self.yaw) >= 90:
            # set linear
            self.linear_msg.data = 0
            self.linear_publisher.publish(self.linear_msg)
        else:
            # set linear
            self.linear_msg.data = LIN_WHEN_ROTATING
            self.linear_publisher.publish(self.linear_msg)

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
                if self.dist[middle_y][middle_x] >= FRONTIER_SKIP_THRESHOLD:
                    self.get_logger().info('[frontierSearch]: point (%d %d) is too far at %f; skipped' % (middle_x, middle_y, self.dist[middle_y][middle_x]))
                    continue

                self.frontierPoints.append((middle_x, middle_y))

            # sort points by distance from current position
            # Current position
            curr_pos = np.array([self.botx_pixel, self.boty_pixel])

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
