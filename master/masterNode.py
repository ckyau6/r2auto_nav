import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8, UInt16, Float64, String, Int8
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
        self.f = 0
        self.l = 0
        self.r = 0

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
        
    def robotControlNode_state_feedback_callback(self, msg):
        self.robotControlNodeState = msg.data
        
    def fsmDebug_callback(self, msg):
        self.state = msg.data
        
    def index_to_angle(self, index, arrLen):
        # return in degrees
        return (index / (arrLen - 1)) * 359
    
    def search_for_wall(self):
        lin_vel1 = Int8()
        lin_vel1.data = 50
        self.linear_publisher.publish(lin_vel1)
        ang_vel1 = Int8()
        ang_vel1.data = -25
        self.anglularVel_publisher.publish(ang_vel1)
        #self.get_logger().info('searching for wall')
    
    def turn_left(self):
        lin_vel2 = Int8()
        lin_vel2.data = 30
        self.linear_publisher.publish(lin_vel2)
        ang_vel2 = Int8()
        ang_vel2.data = 60
        self.anglularVel_publisher.publish(ang_vel2)
        #self.get_logger().info('turning left')
    
    def angle_to_index(self, angle, array):
        length = len(array)
        return int((angle / (360)) * (length - 1))

    def masterFSM(self):
        if self.state == "idle":
            pass

        elif self.state == "wall_following":
            d = 0.4
            d_thres = 0.2
            #d_toofar = 0.6
            
            self.f = self.laser_range[0]
            self.l = self.laser_range[self.angle_to_index(45, self.laser_range)]
            self.r = self.laser_range[self.angle_to_index(315, self.laser_range)]
            self.dr = self.laser_range[self.angle_to_index(270, self.laser_range)]

            #self.get_logger().info(str(self.f))
            '''if self.dr > d_toofar:
                del3 = Float64()
                del3.data = -90.0
                self.deltaAngle_publisher.publish(del3)
                self.get_logger().info('extreme case')'''
            if self.l > d and self.f > d and self.r > d:
                self.search_for_wall()
                self.get_logger().info('1 search for wall')
            elif self.l > d and self.f < d and self.r > d:
                self.turn_left()
                self.get_logger().info('2 slow down n turn left')
            elif self.l > d and self.f > d and self.r < d:
                if self.r < d_thres: 
                    lin_vel3 = Int8()
                    lin_vel3.data = 20
                    self.linear_publisher.publish(lin_vel3)
                    ang_vel3 = Int8()
                    ang_vel3.data = 40
                    self.anglularVel_publisher.publish(ang_vel3)
                    self.get_logger().info('3 dont hit right wall')
            
                else:
                    lin_vel4 = Int8()
                    lin_vel4.data = 100
                    self.linear_publisher.publish(lin_vel4)
                    ang_vel4 = Int8()
                    ang_vel4.data = 0
                    self.anglularVel_publisher.publish(ang_vel4)
                    self.get_logger().info('4 go straighttt')
    
            elif self.l < d and self.f > d and self.r > d:
                self.search_for_wall()
                self.get_logger().info('5 search for wall')
            elif self.l > d and self.f < d and self.r < d:
                self.turn_left()
                self.get_logger().info('6 slow down n turn left')
            elif self.l < d and self.f < d and self.r > d:
                self.turn_left()
                self.get_logger().info('7 slow down n turn left')
            elif self.l < d and self.f < d and self.r < d:
                self.turn_left()
                self.get_logger().info('8 slow down n turn left')
            elif self.l < d and self.f > d and self.r < d:
                lin_vel4 = Int8()
                lin_vel4.data = 100
                self.linear_publisher.publish(lin_vel4)
                ang_vel4 = Int8()
                ang_vel4.data = 0
                self.anglularVel_publisher.publish(ang_vel4)
                self.get_logger().info('9 go straighttt')
                #self.get_logger().info('9')
            else:
                pass        

        else:
            self.state = "idle"


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
