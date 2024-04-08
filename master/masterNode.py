import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8, UInt16, Float64, String, Int8
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import math
import heapq
import random
import scipy.interpolate as si
import sys, threading

import argparse
from collections import deque
#from skimage.morphology import dilation, disk

import time

# used to convert the occupancy grid to an image of map, umpapped, occupied
import scipy.stats
occ_bins = [-1, 0, 50, 100]

# SOME STUFF
lookahead_distance : 0.24 #distance at which the robot will look ahead to determine its next action
speed : 0.18 #maximum velocity of the robot
expansion_size : 3 #extent to which obstacles are expanded in the costmap
target_error : 0.15 #acceptable error margin from the target position
robot_r : 0.2 #safety distance around the robot

pathGlobal = 0

# CLEARANCE_RADIUS is in cm, used to dilate the obstacles
# radius of turtle bot is around 11 cm
CLEARANCE_RADIUS = 10

FRONTIER_THRESHOLD = 5

NAV_TOO_CLOSE = 0.30

BUCKET_TOO_CLOSE = 0.35

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

MAZE_FRONT_RANGE = 20
MAZE_FRONT_LEFT_ANGLE = 0 + MAZE_FRONT_RANGE
MAZE_FRONT_RIGHT_ANGLE = 360 - MAZE_FRONT_RANGE

MAZE_CLEARANCE_ANGLE = 10
MAZE_ROTATE_SPEED = 64

# left, right door and finish line coords in meters from the magic origin
LEFT_DOOR_COORDS_M = (1.20, 2.70)
RIGHT_DOOR_COORDS_M = (1.90, 2.70)
FINISH_LINE_M = 2.10

# return the rotation angle around z axis in degrees (counterclockwise)
def angle_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(t3, t4))

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data = data + [start]
            data = data[::-1]
            return data
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    # If no path to goal was found, return closest path to goal
    if goal not in came_from:
        closest_node = None
        closest_dist = float('inf')
        for node in close_set:
            dist = heuristic(node, goal)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist
        if closest_node is not None:
            data = []
            while closest_node in came_from:
                data.append(closest_node)
                closest_node = came_from[closest_node]
            data = data + [start]
            data = data[::-1]
            return data
    return False

def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]
        N = 2
        t = range(len(x))
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)
        path = [(rx[i],ry[i]) for i in range(len(rx))]
    except:
        path = array
    return path


def pure_pursuit(current_x, current_y, current_heading, path, index):
    global lookahead_distance
    closest_point = None
    v = speed
    for i in range(index,len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        desired_steering_angle = target_heading - current_heading
    else:
        target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path)-1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        v = 0.0
    return v,desired_steering_angle,index

def frontierB(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0.0:
                if i > 0 and matrix[i-1][j] < 0:
                    matrix[i][j] = 2
                elif i < len(matrix)-1 and matrix[i+1][j] < 0:
                    matrix[i][j] = 2
                elif j > 0 and matrix[i][j-1] < 0:
                    matrix[i][j] = 2
                elif j < len(matrix[i])-1 and matrix[i][j+1] < 0:
                    matrix[i][j] = 2
    return matrix

def assign_groups(matrix):
    group = 1
    groups = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 2:
                group = dfs(matrix, i, j, group, groups)
    return matrix, groups

def dfs(matrix, i, j, group, groups):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
        return group
    if matrix[i][j] != 2:
        return group
    if group in groups:
        groups[group].append((i, j))
    else:
        groups[group] = [(i, j)]
    matrix[i][j] = 0
    dfs(matrix, i + 1, j, group, groups)
    dfs(matrix, i - 1, j, group, groups)
    dfs(matrix, i, j + 1, group, groups)
    dfs(matrix, i, j - 1, group, groups)
    dfs(matrix, i + 1, j + 1, group, groups) # sağ alt çapraz
    dfs(matrix, i - 1, j - 1, group, groups) # sol üst çapraz
    dfs(matrix, i - 1, j + 1, group, groups) # sağ üst çapraz
    dfs(matrix, i + 1, j - 1, group, groups) # sol alt çapraz
    return group + 1

def fGroups(groups):
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = [g for g in sorted_groups[:5] if len(g[1]) > 2]    
    return top_five_groups

def calculate_centroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid

def findClosestGroup(matrix,groups, current,resolution,originX,originY):
    targetP = None
    distances = []
    paths = []
    score = []
    max_score = -1 #max score index
    for i in range(len(groups)):
        middle = calculate_centroid([p[0] for p in groups[i][1]],[p[1] for p in groups[i][1]]) 
        path = astar(matrix, current, middle)
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        total_distance = pathLength(path)
        distances.append(total_distance)
        paths.append(path)
    for i in range(len(distances)):
        if distances[i] == 0:
            score.append(0)
        else:
            score.append(len(groups[i][1])/distances[i])
    for i in range(len(distances)):
        if distances[i] > target_error*3:
            if max_score == -1 or score[i] > score[max_score]:
                max_score = i
    if max_score != -1:
        targetP = paths[max_score]
    else: #gruplar target_error*2 uzaklıktan daha yakınsa random bir noktayı hedef olarak seçer. Bu robotun bazı durumlardan kurtulmasını sağlar.
        index = random.randint(0,len(groups)-1)
        target = groups[index][1]
        target = target[random.randint(0,len(target)-1)]
        path = astar(matrix, current, target)
        targetP = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
    return targetP

def pathLength(path):
    for i in range(len(path)):
        path[i] = (path[i][0],path[i][1])
        points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:,0], differences[:,1])
    total_distance = np.sum(distances)
    return total_distance

def costmap(data,width,height,resolution):
    data = np.array(data).reshape(height,width)
    wall = np.where(data == 100)
    for i in range(-expansion_size,expansion_size+1):
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = wall[0]+i
            y = wall[1]+j
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            data[x,y] = 100
    data = data*resolution
    return data

def exploration(data,width,height,resolution,column,row,originX,originY):
        global pathGlobal #Global degisken
        data = costmap(data,width,height,resolution) #Engelleri genislet
        data[row][column] = 0 #Robot Anlık Konum
        data[data > 5] = 1 # 0 olanlar gidilebilir yer, 100 olanlar kesin engel
        data = frontierB(data) #Sınır noktaları bul
        data,groups = assign_groups(data) #Sınır noktaları gruplandır
        groups = fGroups(groups) #Grupları küçükten büyüğe sırala. En buyuk 5 grubu al
        if len(groups) == 0: #Grup yoksa kesif tamamlandı
            path = -1
        else: #Grup varsa en yakın grubu bul
            data[data < 0] = 1 #-0.05 olanlar bilinmeyen yer. Gidilemez olarak isaretle. 0 = gidilebilir, 1 = gidilemez.
            path = findClosestGroup(data,groups,(row,column),resolution,originX,originY) #En yakın grubu bul
            if path != None: #Yol varsa BSpline ile düzelt
                path = bspline_planning(path,len(path)*5)
            else:
                path = -1
        pathGlobal = path
        return

def localControl(scan):
    v = None
    w = None
    for i in range(60):
        if scan[i] < robot_r:
            v = 0.2
            w = -math.pi/4 
            break
    if v == None:
        for i in range(300,360):
            if scan[i] < robot_r:
                v = 0.2
                w = math.pi/4
                break
    return v,w

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

        ''' ================================================ Master FSM ================================================ '''
        self.state = "idle"
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
        self.linear_speed = 100
        # self.yaw_offset = 0
        self.recalc_freq = 30  # frequency to recalculate target angle and fix direction (10 means every one second)
        self.recalc_stat = 0
        
        self.dest_x = []
        self.dest_y = []
        self.path = []
        
        self.lastPlot = time.time()
        self.lastState = time.time()
        
        self.frontierPoints = []
        
        self.robotControlNodeState = ""

    def http_listener_callback(self, msg):
        # "idle", "door1", "door2", "connection error", "http error"
        self.doorStatus = msg.data

    def switch_listener_callback(self, msg):
        # "released" or "pressed"
        self.switchStatus = msg.data
        if self.state == "moving_to_bucket" and self.switchStatus == "pressed":
            self.state = "releasing"
            
            # set linear to be zero
            linear_msg = Int8()
            linear_msg.data = 0
            self.linear_publisher.publish(linear_msg)
            
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
        self.laser_range[self.laser_range==0] = np.nan
        
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
        self.map_data = msg
        self.resolution = self.map_data.info.resolution
        self.originX = self.map_data.info.origin.position.x
        self.originY = self.map_data.info.origin.position.y
        self.width = self.map_data.info.width
        self.height = self.map_data.info.height
        self.data = self.map_data.data
        

    def pos_callback(self, msg):
        # Note: those values are different from the values obtained from odom
        self.x = msg.position.x
        self.y = msg.position.y
        # in degrees (not radians)
        self.yaw = angle_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        # self.yaw += self.yaw_offset
        # self.get_logger().info('x y yaw: %f %f %f' % (self.pos_x, self.pos_y, self.yaw))
        self.get_logger().info(self.x)

    def robotControlNode_state_feedback_callback(self, msg):
        self.robotControlNodeState = msg.data

    def fsmDebug_callback(self, msg):
        self.state = msg.data

    def index_to_angle(self, index, arrLen):
        # return in degrees
        return (index / (arrLen - 1)) * 359
    
    def angle_to_index(self, angle, arrLen):
        # take deg give index
        return int((angle / 359) * (arrLen - 1))
    
    def custom_destroy_node(self):
        # set linear to be zero
        linear_msg = Int8()
        linear_msg.data = 0
        self.linear_publisher.publish(linear_msg)
        
        # set delta angle = 0 to stop
        deltaAngle_msg = Float64()
        deltaAngle_msg.data = 0.0
        self.deltaAngle_publisher.publish(deltaAngle_msg)
        
        self.destroy_node()
    
    def masterFSM(self):
        if self.state == "idle":
            # reset servo to 90, to block ballsssss
            servoAngle_msg = UInt8()
            servoAngle_msg.data = 90
            self.servo_publisher.publish(servoAngle_msg)
            
            # set linear to be zero
            linear_msg = Int8()
            linear_msg.data = 0
            self.linear_publisher.publish(linear_msg)
            
            # set delta angle = 0 to stop
            deltaAngle_msg = Float64()
            deltaAngle_msg.data = 0.0
            self.deltaAngle_publisher.publish(deltaAngle_msg)
            
            # off limit switch
            switch_msg = String()
            switch_msg.data = "deactivate"
            self.switch_publisher.publish(switch_msg)

        elif self.state == "kesif":
            if isinstance(pathGlobal, int) and pathGlobal == 0:
                column = int((self.x - self.originX)/self.resolution)
                row = int((self.y - self.originY)/self.resolution)
                exploration(self.data, self.width, self.height, self.resolution, column, row, self.originX, self.originY)
                self.path = pathGlobal
            else:
                self.path = pathGlobal
            if isinstance(self.path, int) and self.path == -1:
                print("[INFO] EXPLORATION COMPLETED")
                sys.exit()
            self.c = int((self.path[-1][0] - self.originX)/self.resolution)
            self.r = int((self.path[-1][1] - self.originY)/self.resolution)
            self.state = "kesifFalse"
            self.i = 0
            print("[INFO] NEW TARGET SET")
            t = pathLength(self.path)/speed
            t = t - 0.2  # Subtract 0.2 seconds from the calculated time based on the formula x = v * t. The exploration function will be called after t seconds.
            self.t = threading.Timer(t, self.target_callback)  # Calls the exploration function shortly before reaching the target.
            self.t.start()

        # Start of the Path Tracking Block
        elif self.state == "kesifFalse":
            v, w = localControl(self.scan)
            if v == None:
                v, w, self.i = pure_pursuit(self.x, self.y, self.yaw, self.path, self.i)
            if abs(self.x - self.path[-1][0]) < target_error and abs(self.y - self.path[-1][1]) < target_error:
                v = 0.0
                w = 0.0
                self.state = "kesif"
                print("[INFO] REACHED TARGET")
                self.t.join()  # Wait until the thread finishes.
            #twist.linear.x = v
            #twist.angular.z = w

            deltaAngle_msg = Float64()
            deltaAngle_msg.data = math.degrees(w)
            self.deltaAngle_publisher.publish(deltaAngle_msg)
            #self.publisher.publish(twist)
            linear_msg = Int8()
            linear_msg.data = v
            self.linear_publisher.publish(linear_msg)
            time.sleep(0.1)
        # End of the Path Tracking Block
            
        else:
            pass
            
            
'''     elif self.state == "maze_rotating":
            # self.get_logger().info('current yaw: %f' % self.yaw)
            if self.robotControlNodeState == "rotateStop":
                # set linear to start moving forward
                linear_msg = Int8()
                linear_msg.data = self.linear_speed
                self.linear_publisher.publish(linear_msg)
                
                self.state = "maze_moving"
                
                # reset recalc_stat
                self.recalc_stat = 0
                
        elif self.state == "maze_moving":
            # if reached the destination (within one pixel), stop and move to the next destination
            if abs(self.botx_pixel - self.dest_x[0]) <= 1 and abs(self.boty_pixel - self.dest_y[0]) <= 1:
                # set linear to be zero
                linear_msg = Int8()
                linear_msg.data = 0
                self.linear_publisher.publish(linear_msg)
                
                # set delta angle = 0 to stop
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = 0.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)
                
                self.get_logger().info('[maze_moving]: finished moving')
                
                self.dest_x = self.dest_x[1:]
                self.dest_y = self.dest_y[1:]
                
                if len(self.dest_x) == 0:
                    self.state = "idle"
                else:
                    self.move_straight_to(self.dest_x[0], self.dest_y[0])
                return
            
            self.recalc_stat += 1
            
            # recalculate target angle if reach recalc_freq
            # this takes care both for obstacles and re aiming to target coords
            if self.recalc_stat == self.recalc_freq:
                self.recalc_stat = 0
                
                # # if obstacle in front and close to both sides, rotate to move beteween the two
                # if any(self.laser_range[:self.mazeFrontLeftindex] < NAV_TOO_CLOSE) and any(self.laser_range[self.mazeFrontRightindex:] < NAV_TOO_CLOSE):
                    
                #     # find the angle with the shortest distance from 0 to MAZE_FRONT_LEFT_ANGLE
                #     minIndexLeft = np.nanargmin(self.laser_range[:MAZE_FRONT_LEFT_ANGLE])
                #     minAngleleft = self.index_to_angle(minIndexLeft, self.range_len)

                #     # find the angle with the shortest distance from MAZE_FRONT_RIGHT_ANGLE to the end
                #     minIndexRight = np.nanargmin(self.laser_range[MAZE_FRONT_RIGHT_ANGLE:]) + MAZE_FRONT_RIGHT_ANGLE
                #     minAngleRight = self.index_to_angle(minIndexRight, self.range_len)

                #     # target angle will be in between the two angles
                #     targetAngle = (minAngleleft + minAngleRight) / 2
                #     deltaAngle = targetAngle if targetAngle < 180 else targetAngle - 360
                    
                # # else if obstacle in front and close to left, rotate right
                # elif any(self.laser_range[:self.mazeFrontLeftindex] < NAV_TOO_CLOSE):
                    
                #     # find the angle with the shortest distance from 0 to MAZE_FRONT_LEFT_ANGLE
                #     minIndexLeft = np.nanargmin(self.laser_range[:MAZE_FRONT_LEFT_ANGLE])
                #     minAngleleft = self.index_to_angle(minIndexLeft, self.range_len)
                    
                #     # target angle is the angle such that obstacle is no longer in the range of left
                #     # deltaAngle will be the angle diff - MAZE_CLEARANCE_ANGLE
                #     deltaAngle = minAngleleft - MAZE_FRONT_LEFT_ANGLE - MAZE_CLEARANCE_ANGLE
                
                # # else if obstacle in front and close to right, rotate left
                # elif any(self.laser_range[self.mazeFrontRightindex:] < NAV_TOO_CLOSE):
                    
                #     # find the angle with the shortest distance from MAZE_FRONT_RIGHT_ANGLE to the end
                #     minIndexRight = np.nanargmin(self.laser_range[MAZE_FRONT_RIGHT_ANGLE:]) + MAZE_FRONT_RIGHT_ANGLE
                #     minAngleRight = self.index_to_angle(minIndexRight, self.range_len)
                    
                #     # target angle is the angle such that obstacle is no longer in the range of left
                #     # deltaAngle will be the angle diff + MAZE_CLEARANCE_ANGLE
                #     deltaAngle = MAZE_FRONT_RIGHT_ANGLE - minAngleRight + MAZE_CLEARANCE_ANGLE
    
                # # else recalculate target angle for next way point
                # else:
                target_yaw = math.atan2(self.dest_y[0] - self.boty_pixel, self.dest_x[0] - self.botx_pixel) * (180 / math.pi)
                
                deltaAngle = target_yaw - self.yaw
                    
                # set linear to be zero
                linear_msg = Int8()
                linear_msg.data = 0
                self.linear_publisher.publish(linear_msg)
                
                # set delta angle to rotate to target angle
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = deltaAngle * 1.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)
                
                self.state = "maze_rotating"

            # else:
            #     # Calculate the average distance to obstacles on the left, right, and front
            #     left_avg = np.mean(self.laser_range[self.leftIndexL:self.leftIndexH])
            #     right_avg = np.mean(self.laser_range[self.rightIndexL:self.rightIndexH])

            #     anglularVel_msg = Int8()
                
            #     # if both side are too close, move away from the close one
            #     if left_avg < NAV_TOO_CLOSE and right_avg < NAV_TOO_CLOSE:
            #         if left_avg < right_avg:
            #             anglularVel_msg.data = MAZE_ROTATE_SPEED
            #             self.get_logger().info('[maze_moving]: obstacle on left, turning right')
            #         else:
            #             anglularVel_msg.data = -MAZE_ROTATE_SPEED
            #             self.get_logger().info('[maze_moving]: obstacle on right, turning left')
                        
            #     # If there's an obstacle too close on the left, turn right
            #     elif left_avg < NAV_TOO_CLOSE:
            #         anglularVel_msg.data = -MAZE_ROTATE_SPEED
            #         self.get_logger().info('[maze_moving]: obstacle on left, turning right')
                    
            #     # If there's an obstacle too close on the right, turn left
            #     elif right_avg < NAV_TOO_CLOSE:
            #         anglularVel_msg.data = MAZE_ROTATE_SPEED
            #         self.get_logger().info('[maze_moving]: obstacle on right, turning left')
                    
            #     # Otherwise, go straight
            #     else:
            #         anglularVel_msg.data = 0
            #         self.get_logger().info('[maze_moving]: moving forward')

            #     self.anglularVel_publisher.publish(anglularVel_msg)
            
        elif self.state == "http_request":
            if self.doorStatus == "idle":
                # send openDoor request
                door_msg = String()
                door_msg.data = "openDoor"
                self.http_publisher.publish(door_msg)
                self.get_logger().info('[http_request]: opening door')
                
            elif self.doorStatus == "door1":
                self.get_logger().info('[http_request]: door1 opened')
                self.state = "go_to_left_door"
            
            elif self.doorStatus == "door2":
                self.get_logger().info('[http_request]: door2 opened')
                self.state = "go_to_right_door"
                
            elif self.doorStatus == "connection error":
                self.get_logger().info('[http_request]: connection error')
                
            elif self.doorStatus == "http error":
                self.get_logger().info('[http_request]: http error')
            
            else:
                self.get_logger().info('[http_request]: msg error')
                
        elif self.state == "go_to_left_door":
            pass
            
        elif self.state == "go_to_right_door":
            pass
            
        elif self.state == "enter_left_door":
            pass
            
        elif self.state == "enter_right_door":
            pass

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
                linear_msg = Int8()
                linear_msg.data = 0
                self.linear_publisher.publish(linear_msg)
                
                # angle_min > or < 180, the delta angle to move away from the object is still the same
                deltaAngle_msg = Float64()
                deltaAngle_msg.data = angle_min - 180.0
                self.deltaAngle_publisher.publish(deltaAngle_msg)

                self.state = "rotating_to_move_away_from_walls"

            else:
                self.state = "rotating_to_bucket"                   
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
                if any(self.laser_range[0:self.bucketFrontLeftIndex] < BUCKET_TOO_CLOSE) or any(self.laser_range[self.bucketfFrontRightIndex:] < BUCKET_TOO_CLOSE):
                    # infront got something
                    self.get_logger().info('[rotating_to_move_away_from_walls]: something infront')
                    
                    # set linear to be zero
                    linear_msg = Int8()
                    linear_msg.data = 0
                    self.linear_publisher.publish(linear_msg)
                    
                    # set delta angle = 0 to stop
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = 0.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                    
                    # send back to checking_walls_distance to check for all distance again
                    self.state = "checking_walls_distance"
                    
                else:
                    self.get_logger().info('[rotating_to_move_away_from_walls]: front is still clear! go forward')
                    
                    if any(self.laser_range[self.backIndexL:self.backIndexH] < BUCKET_TOO_CLOSE + 0.10):
                        self.get_logger().info('[rotating_to_move_away_from_walls]: butt is still near! go forward')
                        
                        # set linear to be 127 to move forward fastest
                        linear_msg = Int8()
                        linear_msg.data = 127
                        self.linear_publisher.publish(linear_msg)
                    
        
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
                        linear_msg = Int8()
                        linear_msg.data = 0
                        self.linear_publisher.publish(linear_msg)
                        
                        # set delta angle = 0 to stop
                        deltaAngle_msg = Float64()
                        deltaAngle_msg.data = 0.0
                        self.deltaAngle_publisher.publish(deltaAngle_msg)
                        
                        # send back to checking_walls_distance to check for all distance again
                        self.state = "checking_walls_distance"
                    
        elif self.state == "rotating_to_bucket":            
            # if close to forward, go to next state, else allign to bucket first
            if abs(self.bucketAngle) < 2:
                self.get_logger().info('[rotating_to_bucket]: close enough, moving to bucket now')
                self.state = "moving_to_bucket"
            else:
                self.get_logger().info('[rotating_to_bucket]: rotating to face bucket')
                
                if self.bucketAngle < 180:
                    # set linear to be zero 
                    linear_msg = Int8()
                    linear_msg.data = 0
                    self.linear_publisher.publish(linear_msg)
                    
                    # set delta angle = bucketAngle
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = self.bucketAngle * 1.0 # to change int to float type
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                elif self.bucketAngle > 180:
                    # set linear to be zero 
                    linear_msg = Int8()
                    linear_msg.data = 0
                    self.linear_publisher.publish(linear_msg)
                    
                    # set delta angle = bucketAngle -360
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data = self.bucketAngle - 360.0  # to change int to float type
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                else:
                    # the case where it is 180, turn 90 deg first
                    # set linear to be zero 
                    linear_msg = Int8()
                    linear_msg.data = 0
                    self.linear_publisher.publish(linear_msg)
                    
                    # set delta angle = 90
                    deltaAngle_msg = Float64()
                    deltaAngle_msg.data =  90.0
                    self.deltaAngle_publisher.publish(deltaAngle_msg)
                    
                # send to moving_to_bucket to wait for rotation to finish
                self.state = "moving_to_bucket"
                
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
                # set linear to be 127 to move forward fastest
                linear_msg = Int8()
                linear_msg.data = 127
                self.linear_publisher.publish(linear_msg)
                
                # if the bucket is in the to the right, turn left slightly
                anglularVel_msg = Int8()
                
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
                
                # if the bucket is hit, the state transistion and stopping will be done by the switch_listener_callback
            pass
        elif self.state == "releasing":      
            servoAngle_msg = UInt8()
            servoAngle_msg.data = 180
            self.servo_publisher.publish(servoAngle_msg)
            self.get_logger().info('[releasing]: easy clap')
            
            # 5 second after releasing, go back to idle
            if (time.time() - self.lastState) > 5:
                self.state = "idle"
        else:
            mode, tx, ty = map(int, self.state.split())
            mode = int(mode)
            if mode == 0:
                self.dest_x.append(tx)
                self.dest_y.append(ty)
                self.move_straight_to(tx, ty)
            elif mode == 1:
                self.move_to(tx, ty)
            else:
                self.get_logger().info('mode %d does not exist' % mode)
                
                
        ''' #================================================ DEBUG PLOT ================================================ 
'''        
        if self.show_plot and len(self.dilutedOccupancyMap) > 0 and (time.time() - self.lastPlot) > 1:
            # Pixel values
            ROBOT = 0
            UNMAPPED = 1
            OPEN = 2
            OBSTACLE = 3
            MAGIC_ORIGIN = 4
            ESTIMATE_DOOR = 5
            FINISH_LINE = 6
            FRONTIER = 7
            FRONTIER_POINT = 8
            PATH_PLANNING_POINT = 9
            
            # shows the diluted occupancy map with frontiers and path planning points
            self.totalMap = self.dilutedOccupancyMap.copy()
            
            # add padding until certain size, add in the estimated door and finish line incase they exceed for whatever reason
            TARGET_SIZE_M = 5
            TARGET_SIZE_p = max(round(TARGET_SIZE_M / self.map_res), self.leftDoor_pixel[1], self.leftDoor_pixel[0], self.rightDoor_pixel[1], self.rightDoor_pixel[0], self.finishLine_Ypixel)
            
            # Calculate the necessary padding
            padding_height = max(0, TARGET_SIZE_p - self.totalMap.shape[0])
            padding_width = max(0, TARGET_SIZE_p - self.totalMap.shape[1])
            
            # Define the number of pixels to add to the height and width
            padding_height = (0, padding_height)  # Replace with the number of pixels you want to add to the top and bottom
            padding_width = (0, padding_width)  # Replace with the number of pixels you want to add to the left and right

            # Pad the image
            self.totalMap = np.pad(self.totalMap, pad_width=(padding_height, padding_width), mode='constant', constant_values=UNMAPPED)
            
            # Set the value of the door esitmate and finish line, y and x are flipped becasue image coordinates are (row, column)
            self.totalMap[self.leftDoor_pixel[1], self.leftDoor_pixel[0]] = ESTIMATE_DOOR
            self.totalMap[self.rightDoor_pixel[1], self.rightDoor_pixel[0]] = ESTIMATE_DOOR
                        
            self.totalMap[self.finishLine_Ypixel, :] = FINISH_LINE
                           
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
            
            self.lastPlot = time.time()

    def move_straight_to(self, tx, ty):
        target_yaw = math.atan2(ty - self.boty_pixel, tx - self.botx_pixel) * (180 / math.pi)
        self.get_logger().info('currently at (%d %d), moving straight to (%d, %d)' % (self.botx_pixel, self.boty_pixel, tx, ty))
        # self.get_logger().info('currently yaw is %f, target yaw is %f' % (self.yaw, target_yaw))
        deltaAngle = Float64()
        deltaAngle.data = target_yaw - self.yaw
        self.deltaAngle_publisher.publish(deltaAngle)
        self.state = "maze_rotating"

    def find_path_to(self, tx, ty):    
        # unmapped/osbstacle is 0, open space 1
        ok = np.where(self.dilutedOccupancyMap == 2, 1, 0)
        
        # Dijkstra's algorithm
        # get grid coordination
        sx = round((self.pos_x - self.map_origin_x) / self.map_res)
        sy = round((self.pos_y - self.map_origin_y) / self.map_res)
        dist = [[1e18 for x in range(self.map_w)] for y in range(self.map_h)]
        pre = [[(0, 0) for x in range(self.map_w)] for y in range(self.map_h)]
        dist[sy][sx] = 0
        pq = []
        heapq.heappush(pq, (0, sy, sx))
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        while pq:
            d, y, x = heapq.heappop(pq)
            if d > dist[y][x] + 0.001:
                continue
            if y == ty and x == tx:
                break
            for k in range(4):
                ny, nx = y, x
                nd = d + 1.5  # for taking rotation time into account, magical constant
                while True:
                    ny += dy[k]
                    nx += dx[k]
                    nd += 1
                    if ny < 0 or ny >= self.map_h or nx < 0 or nx >= self.map_w:
                        break
                    if ok[ny][nx] == 0:
                        break
                    if dist[ny][nx] > nd:
                        dist[ny][nx] = nd
                        pre[ny][nx] = (y, x)
                        heapq.heappush(pq, (nd, ny, nx))
        self.get_logger().info('distance from cell (%d %d) to cell (%d %d) is %f' % (sx, sy, tx, ty, dist[ty][tx]))
        
        if dist[ty][tx] == 1e18:
            return [], []
        else:
            res_x = []
            res_y = []
            while True:
                res_x.append(tx)
                res_y.append(ty)
                if ty == sy and tx == sx:
                    break
                ty, tx = pre[ty][tx]
            res_x.reverse()
            res_y.reverse()
            return res_x, res_y

    #might need to comment out the part below lmao        
    def move_to(self, tx, ty):
        self.get_logger().info('currently at (%d %d), moving to (%d, %d)' % (self.botx_pixel, self.boty_pixel, tx, ty))
        self.dest_x, self.dest_y = self.find_path_to(tx, ty)
        
        if len(self.dest_x) == 0:
            self.get_logger().info('no path found')
            self.state = "idle"
        else:
            self.get_logger().info('path finding finished')
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
        
        #================================================ Frontier Search ================================================
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
                            if 0 <= i + di < self.map_h and 0 <= j + dj < self.map_w and (i + di, j + dj) in self.frontier:
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

                self.frontierPoints.append((middle_x, middle_y))'''

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
        master_node.destroy_node()

if __name__ == '__main__':
    main()
