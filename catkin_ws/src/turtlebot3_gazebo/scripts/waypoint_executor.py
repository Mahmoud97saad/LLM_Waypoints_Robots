#!/usr/bin/env python3

import rospy
import yaml
from geometry_msgs.msg import Pose, PoseArray, Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf
import math
import json
import time
import os

class WaypointExecutor:
    def __init__(self):
        rospy.init_node('waypoint_executor', anonymous=False, log_level=rospy.INFO)

        # Load environment data from YAML file
        environment_yaml_path = rospy.get_param('~environment_yaml')
        self.load_environment_data(environment_yaml_path)

        # State definitions
        self.STATE_NAVIGATING = 'Navigating'
        self.STATE_EMERGENCY_STOPPED = 'Emergency Stopped'
        self.STATE_REPLANNING = 'Replanning'
        self.STATE_RESUMING_NAVIGATION = 'Resuming Navigation'
        self.STATE_WAITING_FOR_COMMAND = 'Waiting For Command'

        # Initial state
        self.state = self.STATE_WAITING_FOR_COMMAND

        # Emergency Stop Flag
        self.emergency_stop = False

        # Parameters
        self.linear_speed = rospy.get_param('~linear_speed', 0.2)  # meters per second
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)  # radians per second
        self.distance_threshold = rospy.get_param('~distance_threshold', 0.2)  # meters
        self.angle_threshold = rospy.get_param('~angle_threshold', math.radians(5))  # radians (~5 degrees)
        self.obstacle_angle_range = rospy.get_param('~obstacle_angle_range', math.radians(15))  # radians (~15 degrees)
        self.self_obstacle_distance = rospy.get_param('~self_obstacle_distance', 0.3)  # meters
        self.critical_distance = rospy.get_param('~critical_distance', 0.5)  # meters
        # Edit by Shehab: keep planning inflation separate from the raw emergency-stop distance.
        self.obstacle_memory_extra_buffer = rospy.get_param('~obstacle_memory_extra_buffer', 1.25)
        self.waypoint_obstacle_clearance_margin = rospy.get_param(
            '~waypoint_obstacle_clearance_margin',
            self.critical_distance
        )
        # Edit by Shehab: add a body-safety margin for waypoint and segment validation without enlarging
        # the remembered obstacle radius itself.
        self.robot_obstacle_safety_margin = rospy.get_param(
            '~robot_obstacle_safety_margin',
            0.18
        )
        self.max_replans = rospy.get_param('~max_replans', 5)
        self.replan_cooldown = rospy.get_param('~replan_cooldown', 5.0)  # seconds
        self.resume_clear_required = rospy.get_param('~resume_clear_required', 5)
        self.resume_clear_duration = rospy.get_param('~resume_clear_duration', 1.0)
        self.resume_obstacle_clearance_buffer = rospy.get_param('~resume_obstacle_clearance_buffer', 0.15)
        configured_obstacle_memory_radius = rospy.get_param('~obstacle_memory_radius', 1.5)
        self.obstacle_memory_radius = max(
            configured_obstacle_memory_radius,
            self.critical_distance + self.obstacle_memory_extra_buffer
        )
        self.max_remembered_obstacles = rospy.get_param('~max_remembered_obstacles', 20)
        # Edit by Shehab: merge repeated detections of the same physical obstacle into one memory entry.
        self.obstacle_merge_distance = rospy.get_param(
            '~obstacle_merge_distance',
            max(self.obstacle_memory_radius + 0.25, 1.15)
        )
        self.resume_escape_window = rospy.get_param('~resume_escape_window', 2.5)
        self.resume_escape_margin = rospy.get_param('~resume_escape_margin', 0.05)
        self.resume_obstacle_match_tolerance = rospy.get_param('~resume_obstacle_match_tolerance', 0.4)
        # Edit by Shehab: ignore front detections inside the final-goal bubble to prevent false stops
        # caused by the goal-side wall/object cluster.
        self.final_goal_obstacle_ignore_radius = rospy.get_param(
            '~final_goal_obstacle_ignore_radius',
            max(self.critical_distance, self.distance_threshold * 2.0)
        )
        # Edit by Shehab: allow front detections to stay suppressed until the robot physically clears the
        # previous goal area, with a timeout so suppression cannot persist forever.
        self.new_command_goal_clear_grace_duration = rospy.get_param(
            '~new_command_goal_clear_grace_duration',
            2.0
        )
        self.new_command_goal_clear_max_duration = rospy.get_param(
            '~new_command_goal_clear_max_duration',
            12.0
        )
        self.new_command_goal_clear_radius = rospy.get_param(
            '~new_command_goal_clear_radius',
            self.final_goal_obstacle_ignore_radius + 0.1
        )
        self.waypoint_pause = rospy.get_param('~waypoint_pause', 0.1)
        # Edit by Shehab: add odom-based stuck detection so side-edge contacts still trigger replanning
        # even when the front cone does not retrigger.
        self.stuck_detection_window = rospy.get_param('~stuck_detection_window', 2.5)
        self.stuck_distance_threshold = rospy.get_param('~stuck_distance_threshold', 0.08)
        self.stuck_scan_angle_range = rospy.get_param('~stuck_scan_angle_range', math.radians(100))
        self.stuck_scan_distance = rospy.get_param(
            '~stuck_scan_distance',
            max(self.critical_distance + 0.25, 0.65)
        )
        self.replan_attempts = 0
        self.resume_clear_count = 0

        # Subscribers and Publishers
        self.user_command_sub = rospy.Subscriber('/user_command', String, self.user_command_callback)
        self.waypoints_sub = rospy.Subscriber('/llm_waypoints', PoseArray, self.waypoints_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)  # LaserScan subscriber
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Publisher for obstacle data to '/obstacle_data'
        self.obstacle_data_pub = rospy.Publisher('/obstacle_data', String, queue_size=10)

        # Publisher for execution status to '/execution_status'
        self.execution_status_pub = rospy.Publisher('/execution_status', String, queue_size=10)  # New Publisher

        # State variables
        self.current_pose = None
        self.latest_scan = None
        self.waypoints = []
        self.current_waypoint_index = 0
        self.detected_obstacles = []
        self.tf_listener = tf.TransformListener()
        self.navigation_task_active = False
        self.last_goal_position = None
        self.last_completed_goal_position = None
        self.last_goal_reached_time = 0.0
        self.last_command_start_time = 0.0
        self.last_waypoint_execution_start_time = 0.0

        # Cooldown tracking
        self.last_replan_time = 0
        self.last_waypoint_time = rospy.Time.now()
        self.execution_completion_published = False
        self.resume_started_at = 0.0
        self.resume_clear_since = None
        self.motion_monitor_start_time = None
        self.motion_monitor_start_pose = None

        rospy.loginfo("Waypoint Executor initialized.")

    def load_environment_data(self, yaml_file_path):
        try:
            with open(yaml_file_path, 'r') as file:
                self.environment_data = yaml.safe_load(file)
        except Exception as e:
            rospy.logerr(f"Failed to load environment data: {e}")
            self.environment_data = None

    def get_obstacle_boundaries(self):
        if not self.environment_data:
            rospy.logwarn("No environment data loaded. Unable to retrieve obstacle boundaries.")
            return []

        obstacles = [obj for obj in self.environment_data.get('objects', []) if 'Obstacle' in obj.get('name', '')]
        obstacle_boundaries = []
        for obstacle in obstacles:
            pos = obstacle.get('position', {})
            size = obstacle.get('size', {})
            obstacle_boundaries.append({
                'x_min': pos.get('x', 0.0) - size.get('width', 1.0) / 2,
                'x_max': pos.get('x', 0.0) + size.get('width', 1.0) / 2,
                'y_min': pos.get('y', 0.0) - size.get('length', 1.0) / 2,
                'y_max': pos.get('y', 0.0) + size.get('length', 1.0) / 2
            })
        return obstacle_boundaries

    def user_command_callback(self, msg):
        command = msg.data.strip()
        if not command:
            return

        self.prune_obstacles_near_completed_goal()
        self.state = self.STATE_NAVIGATING
        self.emergency_stop = False
        self.replan_attempts = 0
        self.resume_clear_count = 0
        self.last_replan_time = 0
        self.execution_completion_published = False
        self.resume_started_at = 0.0
        self.resume_clear_since = None
        self.motion_monitor_start_time = None
        self.motion_monitor_start_pose = None
        self.navigation_task_active = False
        self.last_command_start_time = time.time()
        self.last_waypoint_execution_start_time = 0.0
        rospy.loginfo(f"Received new navigation command '{command}'. Reset replanning state.")
        self.stop_robot()

    # Edit by Shehab: clear remembered obstacles around the last completed goal so the next mission
    # does not inherit false blocking from the previous stopping area.
    def prune_obstacles_near_completed_goal(self):
        if not self.detected_obstacles:
            return
        if self.last_completed_goal_position is None:
            return

        goal_x = self.last_completed_goal_position['x']
        goal_y = self.last_completed_goal_position['y']
        prune_radius = self.final_goal_obstacle_ignore_radius + self.obstacle_memory_radius + 0.1

        kept_obstacles = []
        removed_count = 0
        for obstacle in self.detected_obstacles:
            distance_to_goal = math.hypot(goal_x - obstacle['x'], goal_y - obstacle['y'])
            if distance_to_goal <= prune_radius:
                removed_count += 1
                continue
            kept_obstacles.append(obstacle)

        if removed_count > 0:
            self.detected_obstacles = kept_obstacles
            rospy.loginfo(
                f"Cleared {removed_count} remembered obstacle(s) near the completed goal at "
                f"({goal_x:.2f}, {goal_y:.2f})."
            )

    def waypoints_callback(self, pose_array):
        rospy.loginfo("Received new waypoints.")

        # Transform and validate waypoints
        valid_waypoints = []

        for pose in pose_array.poses:
            waypoint_pose = PoseStamped()
            waypoint_pose.header = pose_array.header
            waypoint_pose.pose = pose

            try:
                # Transform waypoint to odom frame
                self.tf_listener.waitForTransform("odom", pose_array.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
                transformed_pose = self.tf_listener.transformPose("odom", waypoint_pose)
                waypoint_x = transformed_pose.pose.position.x
                waypoint_y = transformed_pose.pose.position.y

                # Validate only against detected unknown obstacles from live scans.
                is_valid = True
                if is_valid and not self.is_point_clear_of_detected_obstacles(waypoint_x, waypoint_y):
                    is_valid = False
                    rospy.logwarn(
                        f"Waypoint ({waypoint_x:.2f}, {waypoint_y:.2f}) is inside a detected obstacle zone. "
                        "Ignoring this waypoint."
                    )

                if is_valid:
                    valid_waypoints.append((waypoint_x, waypoint_y))
                    rospy.loginfo(f"Transformed and validated waypoint: ({waypoint_x:.2f}, {waypoint_y:.2f})")

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(f"Failed to transform waypoint: {e}")
                continue

        if not valid_waypoints:
            rospy.logerr("No valid waypoints received after validation.")
            return

        if not self.path_segments_clear_of_detected_obstacles(valid_waypoints):
            rospy.logerr("Rejected waypoint path because one or more segments cross a detected obstacle zone.")
            return

        # Update waypoints and reset waypoint index
        self.waypoints = valid_waypoints
        self.current_waypoint_index = 0
        self.navigation_task_active = True
        self.last_waypoint_execution_start_time = time.time()
        final_waypoint = self.waypoints[-1]
        self.last_goal_position = {'x': final_waypoint[0], 'y': final_waypoint[1]}
        rospy.loginfo(f"Waypoints to navigate: {self.waypoints}")

        # Update the last waypoint reception time
        self.last_waypoint_time = rospy.Time.now()
        self.execution_completion_published = False

        # Reset the emergency_stop flag when replanned waypoints arrive after an obstacle stop.
        if self.state in [self.STATE_EMERGENCY_STOPPED, self.STATE_REPLANNING, self.STATE_RESUMING_NAVIGATION]:
            rospy.loginfo("New safe waypoints received. Preparing to resume navigation.")
            self.state = self.STATE_RESUMING_NAVIGATION
            self.emergency_stop = False
            self.resume_clear_count = 0
            self.resume_started_at = time.time()
            self.resume_clear_since = None

        self.motion_monitor_start_time = None
        self.motion_monitor_start_pose = None

    def odom_callback(self, data):  # Callback to update the robot's current pose based on odometry data.
        self.current_pose = data.pose.pose

    def scan_callback(self, data):
        self.latest_scan = data
        if self.current_pose is None:
            return
        if not self.navigation_task_active:
            return
        if self.current_waypoint_index >= len(self.waypoints):
            return
        if self.should_ignore_obstacles_after_new_command():
            return
        if self.should_ignore_obstacles_near_final_goal():
            return

        obstacles_in_map = []
        angle_min = data.angle_min
        angle_increment = data.angle_increment
        num_ranges = len(data.ranges)

        relevant_distances = []
        closest_obstacle = None

        for i in range(num_ranges):
            distance = data.ranges[i]
            if 0.0 < distance < self.critical_distance:
                angle = angle_min + i * angle_increment

                # Check if the obstacle is within the front angular range
                if abs(angle) > self.obstacle_angle_range:
                    continue

                # Ignore obstacles too close to the robot's center (self)
                if distance < self.self_obstacle_distance:
                    continue

                relevant_distances.append(distance)
                if closest_obstacle is None or distance < closest_obstacle['distance']:
                    closest_obstacle = {
                        'distance': distance,
                        'angle': angle
                    }

        # Apply median filtering to reduce noise
        obstacle_detected = False
        if len(relevant_distances) >= 3:
            relevant_distances_sorted = sorted(relevant_distances)
            median_distance = relevant_distances_sorted[len(relevant_distances_sorted) // 2]
            if median_distance < self.critical_distance:
                obstacle_detected = True
        elif len(relevant_distances) > 0:
            min_distance = min(relevant_distances)
            if min_distance < self.critical_distance:
                obstacle_detected = True

        current_time = time.time()

        if obstacle_detected:
            self.resume_clear_count = 0
            self.resume_clear_since = None
            # Edit by Shehab: suppress repeated emergency-stop loops while the robot is already following
            # an escape path away from the same remembered obstacle.
            if self.state == self.STATE_RESUMING_NAVIGATION and self.should_suppress_resume_replan(closest_obstacle):
                return
            if self.state in [self.STATE_NAVIGATING, self.STATE_RESUMING_NAVIGATION] and self.replan_attempts < self.max_replans:
                if (current_time - self.last_replan_time) > self.replan_cooldown:
                    rospy.logwarn("Emergency Stop! Obstacle detected within critical distance.")
                    self.emergency_stop = True
                    self.stop_robot()
                    self.trigger_replan_for_detected_obstacle(closest_obstacle)
        else:
            if self.state == self.STATE_RESUMING_NAVIGATION:
                # Edit by Shehab: require the robot to physically clear the remembered obstacle zone before
                # leaving resume mode.
                if self.robot_inside_remembered_obstacle_zone():
                    self.resume_clear_count = 0
                    self.resume_clear_since = None
                    rospy.loginfo_throttle(
                        1.0,
                        "Holding resume state until the robot clears the remembered obstacle zone."
                    )
                    return

                self.resume_clear_count += 1
                if self.resume_clear_since is None:
                    self.resume_clear_since = current_time
                clear_duration = current_time - self.resume_clear_since
                if (self.resume_clear_count >= self.resume_clear_required and
                        clear_duration >= self.resume_clear_duration):
                    rospy.loginfo("Successfully avoided the obstacle. Resuming navigation.")
                    self.state = self.STATE_NAVIGATING
                    self.resume_clear_count = 0
                    self.resume_started_at = 0.0
                    self.resume_clear_since = None

    def timer_callback(self, event):
        if self.current_pose is None:
            self.stop_robot()
            rospy.logwarn("Current pose is unknown. Awaiting pose data.")
            return

        if not self.navigation_task_active:
            self.stop_robot()
            return

        if self.state in [self.STATE_EMERGENCY_STOPPED, self.STATE_REPLANNING]:
            self.motion_monitor_start_time = None
            self.motion_monitor_start_pose = None
            self.stop_robot()
            return

        if self.current_waypoint_index >= len(self.waypoints):
            self.motion_monitor_start_time = None
            self.motion_monitor_start_pose = None
            self.stop_robot()
            if not self.execution_completion_published:
                self.navigation_task_active = False
                self.state = self.STATE_WAITING_FOR_COMMAND
                self.last_completed_goal_position = dict(self.last_goal_position) if self.last_goal_position else None
                self.last_goal_reached_time = time.time()
                self.publish_execution_completion()
                self.execution_completion_published = True
                rospy.loginfo(
                    "Final target reached. Robot stopped and is waiting for a new command."
                )
            return

        target_x, target_y = self.waypoints[self.current_waypoint_index]
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        # Calculate distance and angle to the waypoint
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Get current yaw
        yaw = self.get_yaw()
        desired_angle = math.atan2(dy, dx)
        angle = desired_angle - yaw

        # Normalize angle to [-pi, pi]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi

        twist = Twist()

        # Proportional control for angular velocity
        if abs(angle) > self.angle_threshold:
            twist.angular.z = self.angular_speed * angle
            twist.linear.x = 0.0  # Stop moving forward while turning
            self.motion_monitor_start_time = None
            self.motion_monitor_start_pose = None
            rospy.logdebug(f"Rotating with angular velocity: {twist.angular.z:.3f} rad/s")
        else:
            # Move at constant linear speed
            twist.linear.x = self.linear_speed
            twist.angular.z = self.angular_speed * angle
            rospy.logdebug(f"Moving forward with linear velocity: {twist.linear.x:.3f} m/s")
            rospy.logdebug(f"Angular velocity set to: {twist.angular.z:.3f} rad/s")

        try:
            self.cmd_vel_pub.publish(twist)
        except rospy.ROSException as e:
            rospy.logdebug(f"Skipping cmd_vel publish during shutdown: {e}")
            return

        if twist.linear.x > 0.0:
            if self.motion_monitor_start_time is None or self.motion_monitor_start_pose is None:
                self.motion_monitor_start_time = time.time()
                self.motion_monitor_start_pose = (current_x, current_y)
            else:
                elapsed = time.time() - self.motion_monitor_start_time
                moved_distance = math.hypot(
                    current_x - self.motion_monitor_start_pose[0],
                    current_y - self.motion_monitor_start_pose[1]
                )
                if moved_distance >= self.stuck_distance_threshold:
                    self.motion_monitor_start_time = time.time()
                    self.motion_monitor_start_pose = (current_x, current_y)
                elif elapsed >= self.stuck_detection_window:
                    # Edit by Shehab: trigger a replan when commanded forward motion produces too little odom
                    # progress, which catches wheel-edge and side-contact stalls.
                    if self.handle_stuck_condition():
                        self.motion_monitor_start_time = None
                        self.motion_monitor_start_pose = None
                        return
                    self.motion_monitor_start_time = time.time()
                    self.motion_monitor_start_pose = (current_x, current_y)

        # Check if waypoint is reached
        if distance < self.distance_threshold:
            rospy.loginfo(f"Waypoint {self.current_waypoint_index + 1} reached at ({target_x:.2f}, {target_y:.2f}).")
            self.current_waypoint_index += 1
            self.motion_monitor_start_time = None
            self.motion_monitor_start_pose = None
            if self.waypoint_pause > 0.0:
                rospy.sleep(self.waypoint_pause)

    # Edit by Shehab: prevent final-goal arrivals from being interrupted by front detections that belong
    # to the target-side boundary or previously handled clutter.
    def should_ignore_obstacles_near_final_goal(self):
        if self.current_pose is None or not self.waypoints:
            return False
        if self.current_waypoint_index != len(self.waypoints) - 1:
            return False

        target_x, target_y = self.waypoints[self.current_waypoint_index]
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        distance_to_goal = math.hypot(target_x - current_x, target_y - current_y)

        if distance_to_goal <= self.final_goal_obstacle_ignore_radius:
            rospy.loginfo_throttle(
                1.0,
                "Ignoring front obstacle detections during final goal approach."
            )
            return True
        return False

    # Edit by Shehab: prevent a new mission from immediately stopping on detections that still belong to
    # the previous goal area before the robot has physically moved away from that cluttered boundary.
    def should_ignore_obstacles_after_new_command(self):
        if self.current_pose is None or self.last_completed_goal_position is None:
            return False
        if self.last_goal_reached_time <= 0.0:
            return False

        grace_anchor_time = self.last_waypoint_execution_start_time or self.last_command_start_time
        if grace_anchor_time <= 0.0:
            return False

        command_age = time.time() - grace_anchor_time
        if command_age > self.new_command_goal_clear_max_duration:
            return False

        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        distance_from_previous_goal = math.hypot(
            current_x - self.last_completed_goal_position['x'],
            current_y - self.last_completed_goal_position['y']
        )

        # Always keep the old short grace period so the robot gets a clean departure even if odom
        # barely changes during the first moments after a new command.
        if command_age <= self.new_command_goal_clear_grace_duration:
            rospy.loginfo_throttle(
                1.0,
                "Ignoring front obstacle detections while clearing the previously reached goal area."
            )
            return True

        # After the initial grace period, keep suppressing only while the robot is still physically
        # inside the previous-goal clearance radius.
        if distance_from_previous_goal > self.new_command_goal_clear_radius:
            return False

        rospy.loginfo_throttle(
            1.0,
            "Ignoring front obstacle detections while clearing the previously reached goal area."
        )
        return True

    def get_yaw(self):
        orientation_q = self.current_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        return yaw

    def estimate_obstacle_position(self, closest_obstacle):
        if closest_obstacle is None or self.current_pose is None:
            return None

        yaw = self.get_yaw()
        obstacle_angle = yaw + closest_obstacle['angle']
        obstacle_x = self.current_pose.position.x + closest_obstacle['distance'] * math.cos(obstacle_angle)
        obstacle_y = self.current_pose.position.y + closest_obstacle['distance'] * math.sin(obstacle_angle)
        rospy.logwarn(
            f"Detected obstacle estimated at ({obstacle_x:.2f}, {obstacle_y:.2f}) "
            f"from range {closest_obstacle['distance']:.2f} m."
        )
        return {
            'x': obstacle_x,
            'y': obstacle_y,
            'radius': self.obstacle_memory_radius,
            'clearance_radius': self.obstacle_memory_radius + self.waypoint_obstacle_clearance_margin
        }

    def find_scan_obstacle(self, max_distance, angle_range=None):
        if self.latest_scan is None:
            return None

        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        closest_obstacle = None

        for index, distance in enumerate(self.latest_scan.ranges):
            if not (0.0 < distance < max_distance):
                continue

            angle = angle_min + index * angle_increment
            if angle_range is not None and abs(angle) > angle_range:
                continue
            if distance < self.self_obstacle_distance:
                continue

            if closest_obstacle is None or distance < closest_obstacle['distance']:
                closest_obstacle = {
                    'distance': distance,
                    'angle': angle
                }

        return closest_obstacle

    def trigger_replan_for_detected_obstacle(self, closest_obstacle):
        obstacle_position_odom = self.estimate_obstacle_position(closest_obstacle)
        obstacle_position_map = self.transform_obstacle_to_map(obstacle_position_odom)
        if obstacle_position_odom:
            self.remember_obstacle(obstacle_position_odom)
            rospy.logwarn(
                f"Emergency stop locked obstacle position at "
                f"({obstacle_position_odom['x']:.2f}, {obstacle_position_odom['y']:.2f}) in odom frame."
            )
        self.state = self.STATE_REPLANNING

        obstacle_data_msg = String()
        if obstacle_position_odom:
            obstacle_data_msg.data = json.dumps({
                'event': 'emergency_stop',
                'obstacle_position': obstacle_position_map or obstacle_position_odom,
                'obstacle_position_odom': obstacle_position_odom,
                'robot_position': {
                    'x': self.current_pose.position.x,
                    'y': self.current_pose.position.y
                },
                'radius': self.obstacle_memory_radius,
                'clearance_radius': self.obstacle_memory_radius + self.waypoint_obstacle_clearance_margin
            })
        else:
            obstacle_data_msg.data = "emergency_stop"
        self.obstacle_data_pub.publish(obstacle_data_msg)

        rospy.loginfo(f"Published obstacle data to /obstacle_data for replanning: {obstacle_data_msg.data}")
        self.last_replan_time = time.time()
        self.replan_attempts += 1

    # Edit by Shehab: fallback recovery when the robot is stuck but the front emergency-stop cone alone
    # is not sufficient to explain the contact.
    def handle_stuck_condition(self):
        if self.current_pose is None:
            return False
        if self.state not in [self.STATE_NAVIGATING, self.STATE_RESUMING_NAVIGATION]:
            return False
        if self.replan_attempts >= self.max_replans:
            return False
        if (time.time() - self.last_replan_time) <= self.replan_cooldown:
            return False

        closest_obstacle = self.find_scan_obstacle(
            self.stuck_scan_distance,
            angle_range=self.stuck_scan_angle_range
        )
        if closest_obstacle is None:
            closest_obstacle = self.find_scan_obstacle(self.stuck_scan_distance, angle_range=None)
        if closest_obstacle is None:
            return False

        rospy.logwarn(
            "Emergency Stop! Robot appears stuck while commanding forward motion; "
            "triggering replanning from nearest scan obstacle."
        )
        self.emergency_stop = True
        self.stop_robot()
        self.trigger_replan_for_detected_obstacle(closest_obstacle)
        return True

    def transform_obstacle_to_map(self, obstacle):
        if obstacle is None:
            return None

        try:
            obstacle_pose = PoseStamped()
            obstacle_pose.header.frame_id = "odom"
            obstacle_pose.header.stamp = rospy.Time(0)
            obstacle_pose.pose.position.x = obstacle['x']
            obstacle_pose.pose.position.y = obstacle['y']
            obstacle_pose.pose.position.z = 0.0
            obstacle_pose.pose.orientation.w = 1.0

            self.tf_listener.waitForTransform("map", "odom", rospy.Time(0), rospy.Duration(0.5))
            transformed_pose = self.tf_listener.transformPose("map", obstacle_pose)
            return {
                'x': transformed_pose.pose.position.x,
                'y': transformed_pose.pose.position.y,
                'radius': obstacle.get('radius', self.obstacle_memory_radius),
                'clearance_radius': obstacle.get(
                    'clearance_radius',
                    obstacle.get('radius', self.obstacle_memory_radius) + self.waypoint_obstacle_clearance_margin
                )
            }
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Could not transform detected obstacle to map frame: {e}")
            return None

    # Edit by Shehab: consolidate repeated scans of the same obstacle by averaging position and keeping
    # the strongest radius/clearance seen so corridor memory does not fragment.
    def remember_obstacle(self, obstacle):
        if 'clearance_radius' not in obstacle:
            obstacle['clearance_radius'] = obstacle.get('radius', self.obstacle_memory_radius) + self.waypoint_obstacle_clearance_margin
        obstacle.setdefault('observations', 1)

        best_match = None
        best_distance = None
        for remembered in self.detected_obstacles:
            distance = math.hypot(remembered['x'] - obstacle['x'], remembered['y'] - obstacle['y'])
            merge_distance = max(
                remembered.get('clearance_radius', remembered.get('radius', 0.0)),
                obstacle.get('clearance_radius', obstacle.get('radius', 0.0)),
                self.obstacle_merge_distance
            )
            if distance <= merge_distance and (best_distance is None or distance < best_distance):
                best_match = remembered
                best_distance = distance

        if best_match is not None:
            previous_observations = int(best_match.get('observations', 1))
            new_observations = previous_observations + 1
            best_match['x'] = (
                best_match['x'] * previous_observations + obstacle['x']
            ) / float(new_observations)
            best_match['y'] = (
                best_match['y'] * previous_observations + obstacle['y']
            ) / float(new_observations)
            best_match['radius'] = max(
                best_match.get('radius', self.obstacle_memory_radius),
                obstacle.get('radius', self.obstacle_memory_radius)
            )
            best_match['clearance_radius'] = max(
                best_match.get('clearance_radius', best_match['radius']),
                obstacle.get('clearance_radius', obstacle.get('radius', self.obstacle_memory_radius))
            )
            best_match['observations'] = new_observations
            return

        self.detected_obstacles.append(obstacle)
        if len(self.detected_obstacles) > self.max_remembered_obstacles:
            self.detected_obstacles = self.detected_obstacles[-self.max_remembered_obstacles:]

    def is_point_clear_of_detected_obstacles(self, x, y):
        for obstacle in self.detected_obstacles:
            radius = obstacle.get('clearance_radius', obstacle.get('radius', self.obstacle_memory_radius))
            radius += self.robot_obstacle_safety_margin
            if math.hypot(x - obstacle['x'], y - obstacle['y']) <= radius:
                return False
        return True

    def path_segments_clear_of_detected_obstacles(self, waypoints):
        if not self.detected_obstacles or self.current_pose is None:
            return True

        segment_points = [(self.current_pose.position.x, self.current_pose.position.y)] + waypoints
        for start, end in zip(segment_points, segment_points[1:]):
            if not self.segment_clear_of_detected_obstacles(start, end):
                return False
        return True

    def segment_clear_of_detected_obstacles(self, start, end, step_size=0.05):
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(math.ceil(distance / step_size)))
        start_distances = [
            math.hypot(start[0] - obstacle['x'], start[1] - obstacle['y'])
            for obstacle in self.detected_obstacles
        ]

        for step in range(steps + 1):
            ratio = step / float(steps)
            x = start[0] + (end[0] - start[0]) * ratio
            y = start[1] + (end[1] - start[1]) * ratio
            for index, obstacle in enumerate(self.detected_obstacles):
                radius = obstacle.get('clearance_radius', obstacle.get('radius', self.obstacle_memory_radius))
                radius += self.robot_obstacle_safety_margin
                obstacle_distance = math.hypot(x - obstacle['x'], y - obstacle['y'])
                start_distance = start_distances[index]

                if start_distance <= radius:
                    if obstacle_distance + 0.02 < start_distance:
                        rospy.logwarn(
                            f"Path segment from ({start[0]:.2f}, {start[1]:.2f}) to "
                            f"({end[0]:.2f}, {end[1]:.2f}) moves deeper into a detected obstacle zone."
                        )
                        return False
                    continue

                if obstacle_distance <= radius:
                    rospy.logwarn(
                        f"Path segment from ({start[0]:.2f}, {start[1]:.2f}) to "
                        f"({end[0]:.2f}, {end[1]:.2f}) crosses a detected obstacle zone near ({x:.2f}, {y:.2f})."
                    )
                    return False
        return True

    def should_suppress_resume_replan(self, closest_obstacle):
        if closest_obstacle is None or self.current_pose is None:
            return False
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return False

        resume_age = time.time() - self.resume_started_at
        if resume_age > self.resume_escape_window:
            return False

        estimated_obstacle = self.estimate_obstacle_position(closest_obstacle)
        if estimated_obstacle is None:
            return False

        matched_obstacle = None
        matched_distance = None
        for obstacle in self.detected_obstacles:
            distance = math.hypot(
                estimated_obstacle['x'] - obstacle['x'],
                estimated_obstacle['y'] - obstacle['y']
            )
            tolerance = max(
                obstacle.get('clearance_radius', obstacle.get('radius', self.obstacle_memory_radius)),
                estimated_obstacle.get('clearance_radius', estimated_obstacle.get('radius', self.obstacle_memory_radius))
            ) + self.resume_obstacle_match_tolerance
            if distance <= tolerance and (matched_distance is None or distance < matched_distance):
                matched_obstacle = obstacle
                matched_distance = distance

        if matched_obstacle is None:
            return False

        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        current_clearance = math.hypot(current_x - matched_obstacle['x'], current_y - matched_obstacle['y'])
        target_clearance = math.hypot(target_x - matched_obstacle['x'], target_y - matched_obstacle['y'])

        if target_clearance <= current_clearance + self.resume_escape_margin:
            return False

        rospy.loginfo_throttle(
            1.0,
            "Suppressing repeated emergency stop while following a replanned escape waypoint away from the same obstacle."
        )
        return True

    def robot_inside_remembered_obstacle_zone(self):
        if self.current_pose is None:
            return False

        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        for obstacle in self.detected_obstacles:
            radius = obstacle.get(
                'clearance_radius',
                obstacle.get('radius', self.obstacle_memory_radius)
            ) + self.resume_obstacle_clearance_buffer
            clearance = math.hypot(current_x - obstacle['x'], current_y - obstacle['y'])
            if clearance <= radius:
                return True
        return False

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        try:
            self.cmd_vel_pub.publish(twist)
        except rospy.ROSException as e:
            rospy.logdebug(f"Skipping stop publish during shutdown: {e}")

    def publish_execution_completion(self):
        completion_msg = String()
        completion_msg.data = "completed"
        self.execution_status_pub.publish(completion_msg)

    def run(self):
        # Timer for continuous waypoint processing
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)  # 10 Hz
        rospy.spin()


if __name__ == '__main__':
    try:
        executor = WaypointExecutor()
        executor.run()
    except rospy.ROSInterruptException:
        pass
