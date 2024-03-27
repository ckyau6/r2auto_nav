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
from PIL import Image
import math
import heapq


# return the rotation angle around z axis in degrees (counterclockwise)
def angle_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(t3, t4))


class MasterNode(Node):
    def __init__(self):
        super().__init__('masterNode')

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
        self.servo_publisher = self.create_publisher(String, 'servoRequest', 10)

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

        ''' ================================================ occupancy map ================================================ '''
        # Create a subscriber to the topic "map"
        self.occ_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occ_callback,
            qos_profile_sensor_data)
        self.occ_subscription  # prevent unused variable warning
        self.occmap = np.array([])
        self.yaw = 0
        self.map_res = 0.05
        self.map_w = self.map_h = 0

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

    def http_listener_callback(self, msg):
        # "idle", "door1", "door2", "connection error", "http error"
        self.doorStatus = msg.data

    def switch_listener_callback(self, msg):
        # "released" or "pressed"
        self.switchStatus = msg.data

    def scan_callback(self, msg):
        # create numpy array to store lidar data
        self.laser_range = np.array(msg.ranges)

        # read min and max range values
        self.range_min = msg.range_min
        self.range_max = msg.range_max

        # replace out of range values with nan
        self.laser_range[self.laser_range < self.range_min] = np.nan
        self.laser_range[self.laser_range > self.range_max] = np.nan

        # store the len since it changes
        self.range_len = len(self.laser_range)

    def bucketAngle_listener_callback(self, msg):
        self.bucketAngle = msg.data

    def occ_callback(self, msg):
        # create numpy array
        self.msgdata = np.array(msg.data)
        # compute histogram to identify percent of bins with -1
        # occ_counts = np.histogram(msgdata, occ_bins)
        # calculate total number of bins
        # total_bins = msg.info.width * msg.info.height
        # log the info
        # self.get_logger().info('Unmapped: %i Unoccupied: %i Occupied: %i Total: %i' % (occ_counts[0][0], occ_counts[0][1], occ_counts[0][2], total_bins))

        # reshape to 2D array using column order
        self.occmap = np.uint8(self.msgdata.reshape(msg.info.height, msg.info.width))
        self.map_res = msg.info.resolution  # according to experiment, should be 0.05 m
        self.map_w = msg.info.width
        self.map_h = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

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
        self.state = msg.data

    def index_to_angle(self, index, arrLen):
        # return in degrees
        return (index / (arrLen - 1)) * 359

    def masterFSM(self):
        if self.state == "idle":
            pass
        elif self.state == "rotating":
            # self.get_logger().info('current yaw: %f' % self.yaw)
            if self.robotControlNodeState == "rotateStop":
                speed = Int8()
                speed.data = self.linear_speed
                self.linear_publisher.publish(speed)
                # self.get_logger().info('start moving forward')
                self.state = "moving"
                self.recalc_stat = 0
        elif self.state == "moving":
            # self.get_logger().info('currently at (%f %f)' % (self.pos_x, self.pos_y))
            if abs(self.pos_x - self.dest_x[0]) < self.map_res and abs(self.pos_y - self.dest_y[0]) < self.map_res:
                speed = Int8()
                speed.data = 0
                self.linear_publisher.publish(speed)
                self.get_logger().info('finished moving')
                self.dest_x = self.dest_x[1:]
                self.dest_y = self.dest_y[1:]
                if len(self.dest_x) == 0:
                    self.state = "idle"
                else:
                    self.move_straight_to(self.dest_x[0], self.dest_y[0])
                return
            self.recalc_stat += 1
            # recalculate target angle
            if self.recalc_stat == self.recalc_freq:
                self.recalc_stat = 0
                target_yaw = math.atan2(self.dest_y[0] - self.pos_y, self.dest_x[0] - self.pos_x) * (180 / math.pi)
                speed = Int8()
                speed.data = 0
                self.linear_publisher.publish(speed)
                deltaAngle = Float64()
                deltaAngle.data = target_yaw - self.yaw
                self.deltaAngle_publisher.publish(deltaAngle)
                self.state = "rotating"
        else:
            mode, tx, ty = map(float, self.state.split())
            mode = int(mode)
            if mode == 0:
                self.dest_x = [tx]
                self.dest_y = [ty]
                self.move_straight_to(tx, ty)
            elif mode == 1:
                self.move_to(tx, ty)
            elif mode == 2:
                self.draw_occ_map_image()
                self.state = "idle"
            else:
                self.get_logger().info('mode %d does not exist' % mode)

    def move_straight_to(self, tx, ty):
        target_yaw = math.atan2(ty - self.pos_y, tx - self.pos_x) * (180 / math.pi)
        self.get_logger().info('currently at (%f %f), moving straight to (%f, %f)' % (self.pos_x, self.pos_y, tx, ty))
        # self.get_logger().info('currently yaw is %f, target yaw is %f' % (self.yaw, target_yaw))
        deltaAngle = Float64()
        deltaAngle.data = target_yaw - self.yaw
        self.deltaAngle_publisher.publish(deltaAngle)
        self.state = "rotating"

    def find_path_to(self, tx, ty):
        ok = [[1 for x in range(self.map_w)] for y in range(self.map_h)]  # 1 if robot can go to that cell
        robot_radius = 3  # avoid going a cell if distance to the nearest occupied cell from that cell is within this value
        occupied_threshold = 50
        for y in range(self.map_h):
            for x in range(self.map_w):
                for i in range(y - robot_radius, y + robot_radius + 1):
                    for j in range(x - robot_radius, x + robot_radius + 1):
                        if 0 <= i < self.map_h and 0 <= j < self.map_w and not (0 <= self.occmap[i][j] <= occupied_threshold):
                            ok[i][j] = 0

        # img = Image.fromarray(np.uint8(ok))
        # plt.imshow(img, cmap='gray', origin='lower')
        # plt.draw_all()
        # plt.pause(0.00000000001)

        # get grid coordination
        sx = round((self.pos_x - self.map_origin_x) / self.map_res)
        sy = round((self.pos_y - self.map_origin_y) / self.map_res)
        tx = round((tx - self.map_origin_x) / self.map_res)
        ty = round((ty - self.map_origin_y) / self.map_res)
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
        res_x = []
        res_y = []
        while True:
            res_x.append(self.map_origin_x + self.map_res * tx)
            res_y.append(self.map_origin_y + self.map_res * ty)
            if ty == sy and tx == sx:
                break
            ty, tx = pre[ty][tx]
        res_x.reverse()
        res_y.reverse()
        return res_x, res_y

    def move_to(self, tx, ty):
        self.get_logger().info('currently at (%f %f), moving to (%f, %f)' % (self.pos_x, self.pos_y, tx, ty))
        self.dest_x, self.dest_y = self.find_path_to(tx, ty)
        self.get_logger().info('path finding finished')
        self.state = "moving"

    def draw_occ_map_image(self):
        data = self.msgdata
        for i in range(len(data)):
            if data[i] == -1:
                data[i] = 0
            else:
                data[i] = 255 - data[i]
        # create image from 2D array using PIL
        img = Image.fromarray(np.uint8(data.reshape(self.map_h, self.map_w)))
        # show the image using grayscale map
        plt.imshow(img, cmap='gray', origin='lower')
        plt.draw_all()
        # pause to make sure the plot gets created
        plt.pause(0.00000000001)


def main(args=None):
    rclpy.init(args=args)

    master_node = MasterNode()

    try:
        rclpy.spin(master_node)
    except KeyboardInterrupt:
        pass
    finally:
        master_node.destroy_node()


if __name__ == '__main__':
    main()
