import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray, Int64MultiArray
from geometry_msgs.msg import Twist, Pose

from nav_msgs.msg import OccupancyGrid

import numpy as np

from scipy.ndimage import generic_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import math


import argparse

PARAMETER_R = 0.93
# use odd number for window size
# WINDOWSIZE = 21
# WINDOWSIZE = 9
WINDOWSIZE = 25

UNMAPPED = 1
OPEN = 2
OBSTACLE = 3

# return the rotation angle around z axis in degrees (counterclockwise)
def angle_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(t3, t4))

class DickStarNode(Node):
    def __init__(self):
        super().__init__('dickStarNode')
        
        ''' ================================================ occupancy map listerner ================================================ '''
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
        
        self.botx_pixel = 0
        self.boty_pixel = 0

        self.magicOriginx_pixel = 0
        self.magicOriginy_pixel = 0
        
        # for dijkstra
        # self.dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        # self.dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        self.dx = np.array([1, 0, -1, 0])
        self.dy = np.array([0, 1, 0, -1])# for dijkstra
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
        
        ''' ================================================ dist map publisher ================================================ '''
        # Create a publisher to the topic "distMap"
        self.distMap_publisher = self.create_publisher(Int64MultiArray, 'distMap', 10)
        
        ''' ================================================ pre map publisher ================================================ '''
        # Create a publisher to the topic "preMap"
        self.preMap_publisher = self.create_publisher(Int64MultiArray, 'preMap', 10)
        
        self.get_logger().info("dickStarNode has started. Aim for the stars, land on the dicks! :D")
        
        ''' ================================================ robot position ================================================ '''
        # Create a subscriber to the topic
        self.pos_subscription = self.create_subscription(
            Pose,
            'position',
            self.pos_callback,
            10)
        self.pos_y = self.pos_x = self.yaw = 0
        
    def pos_callback(self, msg):
        # Note: those values are different from the values obtained from odom
        self.pos_x = msg.position.x
        self.pos_y = msg.position.y
        # in degrees (not radians)
        self.yaw = angle_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        # self.yaw += self.yaw_offset
        # self.get_logger().info('x y yaw: %f %f %f' % (self.pos_x, self.pos_y, self.yaw))
        
    def occ_callback(self, msg):
        occTime = time.time()
        self.get_logger().info('[occ_callback]: new occ map!')
            
        ''' ================================================ Update points ================================================ '''

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
        
        ''' ================================================ Oriori ================================================ '''

        # self.get_logger().info('[occ_callback]: occ_callback took: %s' % timeTaken)
        #
        # TEMPPP
        # Convert the OccupancyGrid to a numpy array
        self.oriorimap = np.array(msg.data, dtype=np.float32).reshape(msg.info.height, msg.info.width)

        ''' ================================================ Processing for Cost Map ================================================ '''
        
        # Create a mask of the UNMAPPED areas
        unmapped_mask = (self.occupancyMap == UNMAPPED)

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

        ''' ================================================ Dick Star ================================================ '''
        
        self.dijkstra()

        timeTaken = time.time() - occTime
        self.get_logger().info('[occ_callback]: occ_callback took: %s' % timeTaken)
        
        ''' ================================================ Publish dist map and pre map ================================================ '''
        # Create a Int64MultiArray message
        msg = Int64MultiArray()
        # Flatten the dist_map and add it to the message
        msg.data = self.dist.flatten()
        # Publish the message
        self.distMap_publisher.publish(msg)
        
        # Create a Int64MultiArray message
        msg = Int64MultiArray()
        # Flatten the pre_map and add it to the message
        msg.data = self.pre.flatten()
        # Publish the message
        self.preMap_publisher.publish(msg)
        
        self.get_logger().info('[occ_callback]: published dist and pre!')
        
        
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
            

def main(args=None):
    rclpy.init(args=args)

    dickStarNode = DickStarNode()

    try:
        rclpy.spin(dickStarNode)
    except KeyboardInterrupt:
        pass
    finally:
        dickStarNode.destroy_node()


if __name__ == '__main__':
    main()
