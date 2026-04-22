#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import json
import math
import time
import itertools
from collections import deque
import heapq

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.current_pose = None
        self.current_yaw = 0.0
        
        # Grid parameters
        self.grid_rows = 5
        self.grid_cols = 5
        
        # Load layout
        try:
            with open('/tmp/gazebo_grid_layout.json', 'r') as f:
                layout = json.load(f)
                self.bonuses = [tuple(b) for b in layout.get('bonuses', [])]
                self.obstacles = [tuple(o) for o in layout.get('obstacles', [])]
        except Exception as e:
            self.get_logger().error(f"Failed to load layout: {e}")
            self.bonuses = []
            self.obstacles = []

        self.start = (0, 0)
        self.goal = (4, 4)
        
        # Wait a bit for Odom messages
        self.get_logger().info("Initializing navigator...")
        time.sleep(2)
        
        # Plan the full route
        self.route = self.plan_route()
        self.get_logger().info(f"Planned route (grid coordinates): {self.route}")
        
        self.path_points = self.expand_path(self.route)
        self.get_logger().info(f"Target world coordinates: {self.path_points}")
        
        self.target_idx = 0
        
        # PID constants
        self.kp_linear = 0.5
        self.kp_angular = 2.0
        self.active = True
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def plan_route(self):
        """Solve TSP for bonuses and goal."""
        best_route = None
        min_dist = float('inf')
        
        # All permutations of bonuses
        for perm in itertools.permutations(self.bonuses):
            full_route = [self.start] + list(perm) + [self.goal]
            dist = 0
            valid = True
            for i in range(len(full_route) - 1):
                path = self.astar(full_route[i], full_route[i+1])
                if path is None:
                    valid = False
                    break
                dist += len(path)
            
            if valid and dist < min_dist:
                min_dist = dist
                best_route = full_route
        
        return best_route if best_route else [self.start, self.goal]

    def astar(self, start, goal):
        """A* pathfinding."""
        if start == goal: return []
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        open_list = []
        heapq.heappush(open_list, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1] # Reverse path
                
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = current[0]+dr, current[1]+dc
                nxt = (nr, nc)
                
                if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and nxt not in self.obstacles:
                    temp_g = g_score[current] + 1
                    if nxt not in g_score or temp_g < g_score[nxt]:
                        g_score[nxt] = temp_g
                        f_score = temp_g + heuristic(nxt, goal)
                        heapq.heappush(open_list, (f_score, nxt))
                        came_from[nxt] = current
        return None

    def expand_path(self, route):
        """Convert grid route to world coordinates."""
        world_path = []
        curr = route[0]
        for next_cell in route[1:]:
            segment = self.astar(curr, next_cell)
            if segment:
                for cell in segment:
                    # In Gazebo, 0,0 grid is top left (-2.0, 2.0). 
                    # X map: col=0 -> -2, col=1 -> -1, ..., col=4 -> 2
                    # Y map: row=0 -> 2, row=1 -> 1, ..., row=4 -> -2
                    world_path.append((float(cell[1] - 2), float(2 - cell[0])))
                curr = next_cell
        return world_path

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        if not self.active:
            return
            
        if self.current_pose is None:
            return
            
        if self.target_idx >= len(self.path_points):
            stop_msg = Twist()
            self.publisher_.publish(stop_msg)
            self.get_logger().info("Goal Reached!")
            self.active = False
            return

        target_x, target_y = self.path_points[self.target_idx]
        dx = target_x - self.current_pose.x
        dy = target_y - self.current_pose.y
        dist = math.sqrt(dx**2 + dy**2)
        
        target_yaw = math.atan2(dy, dx)
        yaw_err = target_yaw - self.current_yaw
        
        # Normalize yaw error [-pi, pi]
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        
        msg = Twist()
        
        # If reached point
        if dist < 0.15:
            self.target_idx += 1
            self.get_logger().info(f"Target {self.target_idx}/{len(self.path_points)} reached")
        else:
            # If yaw error is large, turn towards the goal first before moving
            if abs(yaw_err) > 0.15:
                # Turn with cap to avoid spinning out of control
                msg.angular.z = max(-1.2, min(1.2, self.kp_angular * yaw_err))
                msg.linear.x = 0.0
            else:
                # Move forward, capped linear speed
                msg.linear.x = max(0.05, min(self.kp_linear * dist, 0.25))
                # Slight heading corrections on the move
                msg.angular.z = max(-0.8, min(0.8, self.kp_angular * yaw_err))
            
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigator()
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()