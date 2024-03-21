import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8, Float64, String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
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
        self.map_resolution = 0.05
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
        self.linear_publisher = self.create_publisher(UInt8, 'cmd_linear', 10)
        self.move_command = UInt8()
        self.move_command.data = 1
        self.stop_command = UInt8()
        self.stop_command.data = 0

        ''' ================================================ cmd_angle ================================================ '''
        # Create a publisher to the topic "cmd_angle", which can rotate the robot
        self.angle_publisher = self.create_publisher(Float64, 'cmd_angle', 10)
        self.angle_to_publish = Float64()

        self.get_logger().info("MasterNode has started, bitchesss! >:D")

    def http_listener_callback(self, msg):
        # "idle", "door1", "door2", "connection error", "http error"
        self.doorStatus = msg.data

    def switch_listener_callback(self, msg):
        # "released" or "pressed"
        self.switchStatus = msg.data

    def scan_callback(self, msg):
        # create numpy array
        self.laser_range = np.array(msg.ranges)
        # print to file
        # np.savetxt(scanfile, self.laser_range)
        # replace 0's with nan
        self.laser_range[self.laser_range == 0] = np.nan

    def occ_callback(self, msg):
        # create numpy array
        msgdata = np.array(msg.data)
        # compute histogram to identify percent of bins with -1
        # occ_counts = np.histogram(msgdata, occ_bins)
        # calculate total number of bins
        # total_bins = msg.info.width * msg.info.height
        # log the info
        # self.get_logger().info('Unmapped: %i Unoccupied: %i Occupied: %i Total: %i' % (occ_counts[0][0], occ_counts[0][1], occ_counts[0][2], total_bins))

        # reshape to 2D array using column order
        # self.occdata = np.uint8(oc2.reshape(msg.info.height,msg.info.width,order='F'))
        self.occmap = np.uint8(msgdata.reshape(msg.info.height, msg.info.width))
        # print to file
        # np.savetxt(mapfile, self.occdata)
        self.map_resolution = msg.info.resolution  # according to experiment, should be 0.05 m
        self.map_w = msg.info.width
        self.map_h = msg.info.height

    def pos_callback(self, msg):
        # Note: those values are different from the values obtained from odom
        self.pos_x = int(msg.position.x / self.map_resolution)
        self.pos_y = int(msg.position.y / self.map_resolution)
        # in degrees (not radians)
        self.yaw = angle_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)


    def move_straight_to(self, tx, ty):
        target_yaw = math.atan2(ty - self.pos_y, tx - self.pos_x) * (180 / math.pi)
        self.get_logger().info('currently at (%d %d), moving straight to (%d, %d), target_yaw: %f' % (self.pos_x, self.pos_y, tx, ty, target_yaw))
        self.angle_to_publish.data = target_yaw - self.yaw
        self.angle_publisher.publish(self.angle_to_publish)
        while True:
            rclpy.spin_once(self)
            self.get_logger().info('current yaw: %d' % self.yaw)
            if abs(target_yaw - self.yaw) <= 1:
                break
            time.sleep(0.1)
        time.sleep(2)
        self.get_logger().info('start moving forward')
        self.linear_publisher.publish(self.move_command)
        while True:
            rclpy.spin_once(self)
            self.get_logger().info('current position: %d %d' % (self.pos_x, self.pos_y))
            if tx == self.pos_x and ty == self.pos_y:
                self.linear_publisher.publish(self.stop_command)
                break
            time.sleep(0.1)
        self.get_logger().info('finished moving')

    def find_path_to(self, tx, ty):
        ok = [[True for x in range(self.map_w)] for y in range(self.map_h)]  # True if robot can go to that cell
        robot_radius = 3
        occupied_threshold = 50
        for y in range(self.map_h):
            for x in range(self.map_w):
                for i in range(y - robot_radius, y + robot_radius + 1):
                    for j in range(x - robot_radius, x + robot_radius + 1):
                        if 0 <= i < self.map_h and 0 <= j < self.map_w and self.occmap[i][j] >= occupied_threshold:
                            ok[i][j] = False
        sx = self.pos_x
        sy = self.pos_y
        dist = [[1e18 for x in range(self.map_w)] for y in range(self.map_h)]
        pre = [[(0, 0) for x in range(self.map_w)] for y in range(self.map_h)]
        dist[sy][sx] = 0
        pq = []
        heapq.heappush(pq, (0, sy, sx))
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        while pq:
            d, y, x = heapq.heappop(pq)
            if d > d[y][x] + 0.001:
                continue
            if y == ty and x == tx:
                break
            for k in range(4):
                ny, nx = y, x
                nd = d + 3  # for taking rotation time into account, magical constant
                while True:
                    ny += dy[k]
                    nx += dx[k]
                    nd += 1
                    if ny < 0 or ny >= self.map_h or nx < 0 or nx >= self.map_w:
                        break
                    if not ok[ny][nx]:
                        break
                    if dist[ny][nx] > nd:
                        dist[ny][nx] = nd
                        pre[ny][nx] = (y, x)
                        heapq.heappush(pq, (nd, ny, nx))
        res = []
        while True:
            res.append((tx, ty))
            if ty == sy and tx == sx:
                break
            ty, tx = pre[ty][tx]
        return res

    def move_to(self, tx, ty):
        self.get_logger().info('currently at (%d %d), moving to (%d, %d)' % (self.pos_x, self.pos_y, tx, ty))
        path = self.find_path_to(tx, ty)
        self.get_logger().info('path finding finished')
        for i in range(1, len(path)):
            self.move_to(path[i][0], path[i][1])

    def readKey(self):
        try:
            while True:
                rclpy.spin_once(self)

                x, y = map(int, input("target x and y coordinates: ").split())

                self.move_straight_to(x, y)

        except Exception as e:
            print(e)

        # Ctrl-c detected
        finally:
            self.linear_publisher.publish(self.stop_command)



def main(args=None):
    rclpy.init(args=args)

    master_node = MasterNode()
    master_node.readKey()


if __name__ == '__main__':
    main()
