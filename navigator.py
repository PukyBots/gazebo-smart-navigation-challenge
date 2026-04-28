#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import DeleteEntity
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

        self.start_time = self.get_clock().now()
        self.time_limit = 30.0
        
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
        self.get_logger().info(f"Target world coordinates (len={len(self.path_points)}): {self.path_points}")
        
        if not self.path_points:
            self.get_logger().error("NO VALID PATH FOUND! Obstacles are blocking all possible routes. Please restart.")
            self.active = False
            return
            
        self.target_idx = 0
        
        # PID constants
        self.kp_linear = 3.0
        self.kp_angular = 4.0
        self.active = True
        self.all_bonuses_collected = False
        
        # Startup pause: hold at start point for 5 seconds before moving
        self.startup_delay = 5.0
        self.navigation_started = False
        self.startup_time = None  # Will be set on first odom message (when robot is actually alive)
        
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.bonus_models = []
        for i, (r, c) in enumerate(self.bonuses):
            self.bonus_models.append({
                'name': f'bonus_{i+1}',
                'x': float(c - 2),
                'y': float(2 - r),
                'collected': False
            })
        
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

    def grid_to_world(self, cell):
        """Convert a grid (row, col) cell to Gazebo world (x, y) coordinates."""
        return (float(cell[1] - 2), float(2 - cell[0]))

    def expand_path(self, route):
        """Convert grid route to world coordinates, beginning explicitly from the start."""
        world_path = []
        
        # *** CRITICAL FIX: Always include the start position as the very first waypoint.
        # A* omits the start cell from its returned path, so without this the robot's
        # first target is the SECOND cell — making it look like it starts mid-grid. ***
        start_cell = route[0]
        world_path.append(self.grid_to_world(start_cell))
        
        curr = start_cell
        for next_cell in route[1:]:
            segment = self.astar(curr, next_cell)
            if segment:
                for cell in segment:
                    world_path.append(self.grid_to_world(cell))
                curr = next_cell
        return world_path

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Start the pause timer on the FIRST odom message (robot is now alive in Gazebo)
        if self.startup_time is None:
            self.startup_time = self.get_clock().now()
            self.get_logger().info("Robot detected at start point. Holding position...")

    def control_loop(self):
        if not self.active:
            return
            
        if self.current_pose is None:
            return
            
        current_time = self.get_clock().now()
        
        # --- STARTUP PAUSE ---
        # Hold at the start point for startup_delay seconds before beginning
        if not self.navigation_started:
            startup_elapsed = (current_time - self.startup_time).nanoseconds / 1e9
            remaining = self.startup_delay - startup_elapsed
            if remaining > 0:
                # Publish zero velocity to hold robot still
                self.publisher_.publish(Twist())
                if int(remaining) != getattr(self, '_last_countdown', -1):
                    self._last_countdown = int(remaining)
                    self.get_logger().info(f"Starting in {int(remaining)+1}...")
                return
            else:
                self.navigation_started = True
                self.start_time = self.get_clock().now()  # Reset main timer AFTER pause
                self.get_logger().info("GO! Robot beginning navigation from start point.")
        # ---------------------
        
        elapsed = (current_time - self.start_time).nanoseconds / 1e9
        if elapsed > self.time_limit:
            self.get_logger().warn("Time limit exceeded, stopping the robot.")
            self.handle_timeout()
            return

        # Check if we collected any bonuses
        for b in self.bonus_models:
            if not b['collected']:
                bdx = b['x'] - self.current_pose.x
                bdy = b['y'] - self.current_pose.y
                if math.sqrt(bdx**2 + bdy**2) < 0.25:
                    b['collected'] = True
                    self.get_logger().info(f"Collecting {b['name']}!")
                    self.delete_entity(b['name'])
                    # Check if that was the last one
                    if not self.all_bonuses_collected and all(bm['collected'] for bm in self.bonus_models):
                        self.all_bonuses_collected = True
                        self.time_limit = 30.0
                        self.start_time = self.get_clock().now()
                        self.get_logger().info("All bonuses collected! 30s timer reset for goal arrival")
                    
        if self.target_idx >= len(self.path_points):
            # Only declare goal reached if we're physically near the goal world position (2, -2)
            goal_x, goal_y = self.grid_to_world(self.goal)
            gdx = goal_x - self.current_pose.x
            gdy = goal_y - self.current_pose.y
            if math.sqrt(gdx**2 + gdy**2) < 0.3:
                stop_msg = Twist()
                self.publisher_.publish(stop_msg)
                self.get_logger().info("Goal Reached!")
                self.active = False
            else:
                # Path exhausted but not at goal — layout likely blocked, keep active
                self.get_logger().warn("Path exhausted but not at goal — layout may be blocked. Please restart.")
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
        if dist < 0.12:
            self.target_idx += 1
            self.get_logger().info(f"Target {self.target_idx}/{len(self.path_points)} reached. Robot pose is x={self.current_pose.x:.2f}, y={self.current_pose.y:.2f}")
        else:
            # --- SMOOTH ARC / FLOW MOVEMENT ---
            # Scale forward speed DOWN when angle error is large (don't cut corners sharp)
            # Scale it UP to max when aligned. This creates smooth arcs like a real line follower.
            angle_factor = max(0.0, 1.0 - (abs(yaw_err) / (math.pi / 4)))
            
            msg.linear.x  = max(0.15, min(self.kp_linear * dist, 2.0)) * angle_factor
            msg.angular.z = max(-5.0, min(5.0, self.kp_angular * yaw_err))
            # --------------------------------
            
        self.publisher_.publish(msg)

    def delete_entity(self, name):
        if not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"DeleteEntity service not available to delete {name}")
            return
            
        req = DeleteEntity.Request()
        req.name = name
        self.delete_client.call_async(req)

    def handle_timeout(self):
        self.get_logger().warn('time limit exceeded stopping the robot.')  
        stop_msg= Twist()
        stop_msg.linear.x =0.0
        stop_msg.angular.z =0.0
        self.publisher_.publish(stop_msg)
        self.active = False
         
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
