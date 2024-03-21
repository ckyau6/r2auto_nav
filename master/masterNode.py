import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8, Float64, String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
import time


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
        self.occdata = np.array([])
        self.yaw = 0

        ''' ================================================ robot position ================================================ '''
        # Create a subscriber to the topic
        self.pos_subscription = self.create_subscription(
            Pose,
            'position',
            self.pos_callback,
            10)

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

        def angle_to_index(angle, array):
            length = len(array)
            return int((angle / (360)) * (length - 1))
        
        self.f = self.laser_range[0]
        self.l = self.laser_range[angle_to_index(45, self.laser_range)]
        self.r = self.laser_range[angle_to_index(315, self.laser_range)]

    def occ_callback(self, msg):
        # create numpy array
        msgdata = np.array(msg.data)
        # compute histogram to identify percent of bins with -1
        # occ_counts = np.histogram(msgdata, occ_bins)
        # calculate total number of bins
        # total_bins = msg.info.width * msg.info.height
        # log the info
        # self.get_logger().info('Unmapped: %i Unoccupied: %i Occupied: %i Total: %i' % (occ_counts[0][0], occ_counts[0][1], occ_counts[0][2], total_bins))

        # make msgdata go from 0 instead of -1, reshape into 2D
        oc2 = msgdata + 1
        # reshape to 2D array using column order
        # self.occdata = np.uint8(oc2.reshape(msg.info.height,msg.info.width,order='F'))
        self.occdata = np.uint8(oc2.reshape(msg.info.height, msg.info.width))
        # print to file
        # np.savetxt(mapfile, self.occdata)
        self.map_res = msg.info.resolution  # according to experiment, should be 0.05 m

    def pos_callback(self, msg):
        # Note: those values are different from the values obtained from odom
        self.pos_x = msg.position.x
        self.pos_y = msg.position.y
        # in degrees (not radians)
        self.yaw = angle_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        self.get_logger().info('x y yaw: %f %f %f' % (self.pos_x, self.pos_y, self.yaw))

    def wall_follow(self):
        #Getting distance data
        self.get_logger().info("Wall Follow function started")
        d = 0.20
        d_thres = 0.05

        def search_for_wall():
            self.linear_publisher.publish(self.move_command)
            time.sleep(3)
            self.angle_to_publish.data = -10
            self.angle_publisher.publish(self.angle_to_publish.data)
        
        def turn_left():
            self.angle_to_publish.data = 20
            self.angle_publisher.publish(self.angle_to_publish.data)

        if self.l > d and self.f < d and self.r > d:
            search_for_wall()
        elif self.l < d and self.f > d and self.r > d:
            turn_left()
        elif self.l > d and self.f < d and self.r < d:
            if self.f < d_thres: #If we change to lin + ang vel, change f to r
                turn_left()
            else:
                self.linear_publisher.publish(1)
        elif self.l < d and self.f > d and self.r > d:
            search_for_wall()
        elif self.l > d and self.f < d and self.r < d:
            turn_left()
        elif self.l < d and self.f < d and self.r > d:
            turn_left()
        elif self.l < d and self.f < d and self.r < d:
            turn_left()
        elif self.l < d and self.f > d and self.r < d:
            search_for_wall()
        else:
            pass

def main(args=None):
    rclpy.init(args=args)

    master_node = MasterNode()

    try:
        rclpy.spin(master_node)
        master_node.wall_follow()
    except KeyboardInterrupt:
        pass
    finally:
        master_node.destroy_node()


if __name__ == '__main__':
    main()
