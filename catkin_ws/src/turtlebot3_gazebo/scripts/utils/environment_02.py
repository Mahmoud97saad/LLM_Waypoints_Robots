#!/usr/bin/env python3

import rospy
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math

MAX_SHORT_WAYPOINTS = 5
MIN_LONG_WAYPOINTS = 6
MAX_LONG_WAYPOINTS = 10


def calculate_waypoint_count(current_position, target_position, waypoint_spacing):
    distance = math.hypot(
        target_position['x'] - current_position['x'],
        target_position['y'] - current_position['y']
    )
    spacing = max(float(waypoint_spacing), 0.1)
    raw_waypoint_count = max(1, int(math.ceil(distance / spacing)))
    if raw_waypoint_count < MIN_LONG_WAYPOINTS:
        waypoint_count = min(raw_waypoint_count, MAX_SHORT_WAYPOINTS)
    else:
        waypoint_count = min(raw_waypoint_count, MAX_LONG_WAYPOINTS)
    return distance, waypoint_count


def enforce_waypoint_count(waypoints, waypoint_count):
    if len(waypoints) <= waypoint_count:
        return waypoints

    if waypoint_count <= 1:
        rospy.logwarn(
            f"LLM returned {len(waypoints)} waypoint(s); keeping only the final target waypoint."
        )
        return [waypoints[-1]]

    last_index = len(waypoints) - 1
    step = last_index / float(waypoint_count - 1)
    sampled_waypoints = [waypoints[int(round(index * step))] for index in range(waypoint_count)]
    sampled_waypoints[-1] = waypoints[-1]
    rospy.logwarn(
        f"LLM returned {len(waypoints)} waypoint(s); reducing to {waypoint_count} waypoint(s)."
    )
    return sampled_waypoints


def _point_in_any_bounds(point, corridor_bounds):
    for corridor_name, bounds in corridor_bounds.items():
        if (bounds['x_min'] <= point['x'] <= bounds['x_max'] and
                bounds['y_min'] <= point['y'] <= bounds['y_max']):
            return corridor_name
    return None


def _segment_stays_in_free_space(start, end, corridor_bounds, step_size=0.05):
    distance = math.hypot(end['x'] - start['x'], end['y'] - start['y'])
    steps = max(1, int(math.ceil(distance / step_size)))

    for step in range(steps + 1):
        ratio = step / float(steps)
        sampled_point = {
            'x': start['x'] + (end['x'] - start['x']) * ratio,
            'y': start['y'] + (end['y'] - start['y']) * ratio
        }
        if _point_in_any_bounds(sampled_point, corridor_bounds) is None:
            rospy.logwarn(
                f"Path segment from ({start['x']:.2f}, {start['y']:.2f}) to "
                f"({end['x']:.2f}, {end['y']:.2f}) leaves free corridor space near "
                f"({sampled_point['x']:.2f}, {sampled_point['y']:.2f})."
            )
            return False
    return True


def validate_waypoints(waypoints, environment_data, safe_margin, target_position, current_position=None):
    main_corridor_width = environment_data.get('main_corridor_width', 4.0)
    main_corridor_length = environment_data.get('main_corridor_length', 10.0)
    side_corridor_length = environment_data.get('side_corridor_length', 4.0)
    side_corridor_width = environment_data.get('side_corridor_width', 18.0)

    # Calculate boundaries
    main_corridor_x_min = -main_corridor_width / 2 + safe_margin  # e.g., -1.5
    main_corridor_x_max = main_corridor_width / 2 - safe_margin  # e.g., 1.5
    main_corridor_y_min = 0.0  # 0.0
    main_corridor_y_max = main_corridor_length - safe_margin  # 9.5

    side_corridor_x_min = -side_corridor_width / 2 + safe_margin  # e.g., -8.5
    side_corridor_x_max = side_corridor_width / 2 - safe_margin  # e.g., 8.5
    side_corridor_y_min = main_corridor_length + safe_margin  # 10.5
    side_corridor_y_max = main_corridor_length + side_corridor_length - 2  # 12.0

    corridor_bounds = {
        'Main_Corridor': {
            'x_min': main_corridor_x_min,
            'x_max': main_corridor_x_max,
            'y_min': main_corridor_y_min,
            'y_max': main_corridor_y_max
        },
        'Side_Corridor': {
            'x_min': side_corridor_x_min,
            'x_max': side_corridor_x_max,
            'y_min': side_corridor_y_min,
            'y_max': side_corridor_y_max
        },
        'Junction': {
            'x_min': main_corridor_x_min,
            'x_max': main_corridor_x_max,
            'y_min': main_corridor_y_max,
            'y_max': side_corridor_y_min
        }
    }

    rospy.loginfo("Validating waypoints within corridor boundaries:")
    for corridor_name, bounds in corridor_bounds.items():
        rospy.loginfo(
            f"{corridor_name}: X-axis {bounds['x_min']} <= x <= {bounds['x_max']}, "
            f"Y-axis {bounds['y_min']} <= y <= {bounds['y_max']}"
        )

    for waypoint in waypoints:
        x = waypoint['x']
        y = waypoint['y']

        corridor_name = _point_in_any_bounds(waypoint, corridor_bounds)
        if corridor_name is None:
            rospy.logwarn(f"Waypoint ({x}, {y}) is outside corridor boundaries.")
            return None
        rospy.loginfo(f"Waypoint ({x}, {y}) is within {corridor_name}.")

    if current_position is not None:
        segment_points = [current_position] + waypoints
        for start, end in zip(segment_points, segment_points[1:]):
            if not _segment_stays_in_free_space(start, end, corridor_bounds):
                return None

    # Ensure the final waypoint matches the target position within a tolerance
    final_waypoint = waypoints[-1]
    try:
        target_x = target_position['x']
        target_y = target_position['y']
    except (TypeError, KeyError) as e:
        rospy.logerr(f"Invalid target_position format: {e}")
        return None

    tolerance = 0.05  # meters

    distance = ((final_waypoint['x'] - target_x) ** 2 + (final_waypoint['y'] - target_y) ** 2) ** 0.5
    rospy.loginfo(f"Distance between final waypoint and target: {distance} meters")

    if distance >= tolerance:
        rospy.logerr(
            f"Final waypoint ({final_waypoint['x']}, {final_waypoint['y']}) does not match the target position ({target_x}, {target_y}) within a tolerance of {tolerance} meters."
        )
        return None

    return waypoints


def agents(llm_client, system_prompt, user_prompt):
    try:
        response = llm_client.chat(
            model='llama3.1', # qwen2.5, llama3.1, mathstral
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response['message']['content']
        rospy.loginfo(f"LLM Response:\n{answer}")
        return answer
    except Exception as e:
        rospy.logerr(f"LLM interaction failed: {e}")
        return None


def parse_waypoints(response):
    if not response:
        rospy.logerr("Empty response received for waypoints parsing.")
        return None
    if not isinstance(response, str):
        rospy.logerr("Response is not a string.")
        return None
    try:
        # Remove code fences
        response = re.sub(r'^```(json)?\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response.strip())

        # Replace single quotes with double quotes
        response = response.replace("'", '"')

        # Extract the JSON array
        match = re.search(r'\[.*]', response, re.DOTALL)
        if not match:
            rospy.logerr("LLM response does not contain a valid JSON array.")
            return None
        json_str = match.group(0)

        # Parse the JSON string
        waypoints = json.loads(json_str)

        # Validate waypoints
        if not isinstance(waypoints, list):
            rospy.logerr("Waypoints should be a list of dictionaries.")
            return None
        for waypoint in waypoints:
            if not isinstance(waypoint, dict) or 'x' not in waypoint or 'y' not in waypoint:
                rospy.logerr("Invalid waypoint format. Each waypoint must be a dictionary with 'x' and 'y' keys.")
                return None
            # Convert coordinates to float
            waypoint['x'] = float(waypoint['x'])
            waypoint['y'] = float(waypoint['y'])
        return waypoints
    except Exception as e:
        rospy.logerr(f"Failed to parse waypoints: {e}")
        return None


def determine_corridor(position):
    # corridor boundaries
    corridors = {
        'Main_Corridor': {
            'x_min': -1.5,
            'x_max': 1.5,
            'y_min': 0.0,
            'y_max': 9.5
        },
        'Side_Corridor': {
            'x_min': -8.5,
            'x_max': 8.5,
            'y_min': 10.5,
            'y_max': 12.0
        }
    }

    x = position['x']
    y = position['y']

    # rospy.loginfo(f"Checking if position ({x}, {y}) is within any corridor.")
    for corridor_name, bounds in corridors.items():
        # rospy.loginfo(f"Checking {corridor_name}:")
        # rospy.loginfo(f"  X-axis: {bounds['x_min']} <= {x} <= {bounds['x_max']}")
        # rospy.loginfo(f"  Y-axis: {bounds['y_min']} <= {y} <= {bounds['y_max']}")
        if (bounds['x_min'] <= x <= bounds['x_max'] and
                bounds['y_min'] <= y <= bounds['y_max']):
            rospy.loginfo(f"Position ({x}, {y}) is in {corridor_name}.")
            return corridor_name
    rospy.logwarn(f"Position ({x}, {y}) does not lie within any defined corridor.")
    return None


def plot_waypoints(current_position, target_position, waypoints, environment_data, safe_margin, junction_point=None):
    main_corridor_width = environment_data.get('main_corridor_width', 4.0)
    main_corridor_length = environment_data.get('main_corridor_length', 10.0)
    side_corridor_length = environment_data.get('side_corridor_length', 4.0)
    side_corridor_width = environment_data.get('side_corridor_width', 18.0)

    main_corridor_x_min = -main_corridor_width / 2 + safe_margin
    main_corridor_x_max = main_corridor_width / 2 - safe_margin
    main_corridor_y_min = 0.0
    main_corridor_y_max = main_corridor_length - safe_margin

    side_corridor_x_min = -side_corridor_width / 2 + safe_margin
    side_corridor_x_max = side_corridor_width / 2 - safe_margin
    side_corridor_y_min = main_corridor_length + safe_margin
    side_corridor_y_max = main_corridor_length + side_corridor_length - 2

    rospy.loginfo("Plotting waypoints and corridor boundaries.")

    # Set fixed figure size (e.g., 10 inches by 8 inches)
    fig, ax = plt.subplots(figsize=(7, 6))  # Adjust the size as needed

    # Draw Main Corridor
    main_corridor = plt.Rectangle((main_corridor_x_min, main_corridor_y_min),
                                  main_corridor_x_max - main_corridor_x_min,
                                  main_corridor_y_max - main_corridor_y_min,
                                  edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Main Corridor')
    ax.add_patch(main_corridor)

    # Draw Side Corridor
    side_corridor = plt.Rectangle((side_corridor_x_min, side_corridor_y_min),
                                  side_corridor_x_max - side_corridor_x_min,
                                  side_corridor_y_max - side_corridor_y_min,
                                  edgecolor='green', facecolor='lightgreen', alpha=0.5, label='Side Corridor')
    ax.add_patch(side_corridor)

    # Plot current and target positions
    ax.plot(current_position['x'], current_position['y'], 'ro', label='Current Position')
    ax.plot(target_position['x'], target_position['y'], 'go', label='Target Position')

    # Plot waypoints
    if waypoints:
        x_waypoints = [wp['x'] for wp in waypoints]
        y_waypoints = [wp['y'] for wp in waypoints]
        # Plot the connection from current position to the first waypoint
        ax.plot([current_position['x'], x_waypoints[0]], [current_position['y'], y_waypoints[0]], 'k--', label='Path')
        ax.plot(x_waypoints, y_waypoints, 'k--', marker='x', label='Waypoints')

    # Plot Junction Point if it exists
    if junction_point:
        ax.plot(junction_point[0], junction_point[1], 'mo', label='Junction Point')

    ax.set_xlabel('X-axis (meters)', fontweight='bold')
    ax.set_ylabel('Y-axis (meters)', fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    plt.axis('equal')
    save_dir = '/home/albert/Pictures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, '3.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)



def generate_waypoints_with_navigation_agent(llm_client, current_pose, target_object, environment_data, safe_margin,
                                             waypoint_spacing):
    max_attempts = 3  # Maximum number of attempts per agent

    # Extract corridor dimensions
    main_corridor_width = environment_data.get('main_corridor_width', 4.0)
    main_corridor_length = environment_data.get('main_corridor_length', 10.0)
    side_corridor_length = environment_data.get('side_corridor_length', 4.0)
    side_corridor_width = environment_data.get('side_corridor_width', 18.0)

    # Calculate boundaries
    main_corridor_x_min = -main_corridor_width / 2 + safe_margin  # e.g., -1.5
    main_corridor_x_max = main_corridor_width / 2 - safe_margin  # e.g., 1.5
    main_corridor_y_min = 0.0  # 0.0
    main_corridor_y_max = main_corridor_length - safe_margin  # 9.5

    side_corridor_x_min = -side_corridor_width / 2 + safe_margin  # e.g., -8.5
    side_corridor_x_max = side_corridor_width / 2 - safe_margin  # e.g., 8.5
    side_corridor_y_min = main_corridor_length + safe_margin  # 10.5
    side_corridor_y_max = main_corridor_length + side_corridor_length - 2  # 12.0

    # Calculate junction point
    junction_point_1 = (0.0, 10.5)

    # Determine robot's current corridor
    current_position = {
        'x': current_pose.pose.position.x,
        'y': current_pose.pose.position.y
    }
    rospy.loginfo(f"Determining current corridor for position ({current_position['x']}, {current_position['y']}).")
    current_corridor = determine_corridor(current_position)

    if current_corridor is None:
        rospy.logerr("Cannot determine the current corridor of the robot.")
        # return

    # Determine which corridor the target is in
    target_x = target_object['position']['x']
    target_y = target_object['position']['y']

    rospy.loginfo(f"Determined Target Position: x={target_x}, y={target_y}")

    target_position = {
        'x': target_x,
        'y': target_y
    }

    robot_to_target_distance, waypoint_count = calculate_waypoint_count(
        current_position,
        target_position,
        waypoint_spacing
    )
    rospy.loginfo(
        f"Robot-to-target distance is {robot_to_target_distance:.2f} meters; "
        f"requesting {waypoint_count} waypoint(s)."
    )

    target_corridor = determine_corridor(target_position)

    if target_corridor is None:
        rospy.logerr("Target position does not lie within any defined corridor.")
        junction_point = None
    else:
        if current_corridor == target_corridor:
            junction_point = None
            rospy.loginfo("Robot and target are in the same corridor. No junction point required.")
        else:
            # Assign junction point based on the current corridor and target corridor
            if current_corridor == 'Main_Corridor':
                if target_corridor == 'Side_Corridor':
                    junction_point = junction_point_1
                else:
                    rospy.logwarn(f"No junction point defined for target corridor: {target_corridor}.")
                    junction_point = None
            elif current_corridor == 'Side_Corridor':
                if target_corridor == 'Main_Corridor':
                    junction_point = junction_point_1
                else:
                    rospy.logwarn(f"No junction point defined for target corridor: {target_corridor}.")
                    junction_point = None
            elif current_position != 'Main_Corridor':
                junction_point = junction_point_1
            elif current_position != 'Side_Corridor':
                junction_point = junction_point_1

            else:
                rospy.logwarn(f"No junction point defined for current corridor: {current_corridor}.")
                junction_point = None

    # Handle junction point string
    if junction_point:
        junction_point_str = f"{junction_point[0]:.2f}, {junction_point[1]:.2f}"
        rospy.loginfo(f"Using Junction Point: {junction_point_str}")
        waypoint_count = max(waypoint_count, 2)
    else:
        junction_point_str = "None"
        rospy.loginfo("No Junction Point required for this navigation.")

    # Construct system and user prompts
    navigation_system_prompt = f"""
Robot's current position: ({current_position['x']:.2f}, {current_position['y']:.2f})
Target position: ({target_x:.2f}, {target_y:.2f})
Corridor boundaries:

Main Corridor:
- X-axis: from {main_corridor_x_min:.2f} to {main_corridor_x_max:.2f}
- Y-axis: from {main_corridor_y_min:.2f} to {main_corridor_y_max:.2f}

Side Corridor:
- X-axis: from {side_corridor_x_min:.2f} to {side_corridor_x_max:.2f}
- Y-axis: from {side_corridor_y_min:.2f} to {side_corridor_y_max:.2f}

Safe margin: {safe_margin} meters from walls
Robot-to-target distance: {robot_to_target_distance:.2f} meters
Waypoint spacing target: {waypoint_spacing} meters
Required waypoint count: {waypoint_count}

You are a navigation assistant for a mobile robot in a T-shaped corridor map with two corridors: Main Corridor and Side_Corridor.
Provide a concise sequence of waypoints as (x, y) coordinates for the robot to follow to reach the destination, avoiding obstacles and maintaining the safe margins defined above. Ensure that:
1. All waypoints are within the corridor boundaries defined above.
2. Help in creating a list of waypoints from the robot's current position to the target position in the best sequence on both axis.
3. Room_number_plates from 101 to 106 are in Main_Corridor and Room_number_plates 107 to 110 and both windows are in Side_Corridor.
4. If the robot needs to move from one corridor to another, must include the junction point ({junction_point_str}) in the waypoints as the first waypoint and the next waypoint must be in the corridor where the target ({target_position}) is located.
5. Ensure the path is direct and efficient, facilitating smooth transitions between corridors when necessary.
6. Generate exactly {waypoint_count} waypoint(s), based on the robot-to-target distance and waypoint spacing. Short routes must use fewer than 6 waypoints; long routes must use between 6 and 10 waypoints, never more than 10.
7. Do not include any comments, annotations, or additional text. The output should be a valid JSON array of waypoints only.
8. Format the output strictly as a JSON array of coordinates without any comments or additional text.
9. Please follow all rules it's a humble request.
"""

    navigation_user_prompt = f"""

Robot's current position: ({current_position['x']:.2f}, {current_position['y']:.2f})
Target position: ({target_x:.2f}, {target_y:.2f})

Safe margin: {safe_margin} meters from walls
Robot-to-target distance: {robot_to_target_distance:.2f} meters
Waypoint spacing target: {waypoint_spacing} meters
Required waypoint count: {waypoint_count}

Please provide a concise sequence of waypoints as (x, y) coordinates for the robot to follow to reach the destination, avoiding obstacles and maintaining
the safe margins defined above. Ensure that:
1. The waypoints form a continuous and logical path towards the destination that reduces the distance between the robot's current position and the target object without any backtracking.
2. Generate exactly {waypoint_count} waypoint(s), including the final target waypoint. This count is computed from the robot-to-target distance; short routes must produce fewer than 6 waypoints, while long routes must produce between 6 and 10 waypoints and never more than 10.
3. **All waypoints must form a straight or smoothly curved path towards the target without deviating away**.
4. The final waypoint must be exactly at the target object's position.
5. Do not include the current position in the waypoints.
6. Each waypoint should be about {waypoint_spacing} meters apart from the previous one to avoid redundancy; the final target waypoint may be closer if needed to end exactly at the target.
7. Ensure that the X-axis and Y-axis values of all waypoints remain within the corridor's boundaries without deviation.
8. If the robot needs to move from one corridor to another, must include the junction point ({junction_point_str}) in the waypoints as the first waypoint; otherwise, generate waypoints normally as you are generating.
9. Please follow all rules it's a humble request.

Do not include any comments, annotations, or additional text. The output should be a valid JSON array of waypoints only.

Format the output strictly as a JSON array of coordinates without any comments or additional text.

Example:
[
  {{"x": 1.25, "y": 2.63}},
  {{"x": 1.25, "y": 5.36}},
  {{"x": 1.25, "y": 7.86}}
]

Waypoints:
"""

    for attempt in range(max_attempts):
        rospy.loginfo(f"Attempt {attempt + 1} to generate waypoints.")
        navigation_output = agents(llm_client, navigation_system_prompt, navigation_user_prompt)
        if navigation_output:
            waypoints = parse_waypoints(navigation_output)
            if waypoints:
                waypoints = enforce_waypoint_count(waypoints, waypoint_count)
                # Validate waypoints here
                waypoints = validate_waypoints(
                    waypoints,
                    environment_data,
                    safe_margin,
                    target_object['position'],
                    current_position
                )
                if waypoints:
                    rospy.loginfo(f"Navigation Agent succeeded on attempt {attempt + 1}")
                    # Optional: Visualize waypoints
                    plot_waypoints(current_position, target_position, waypoints, environment_data, safe_margin)
                    return waypoints
        rospy.logwarn(f"Navigation Agent attempt {attempt + 1} failed. Retrying...")
    else:
        rospy.logerr("Navigation Agent failed after maximum attempts.")
        return None


if __name__ == '__main__':
    try:
        # Initialize ROS node
        rospy.init_node('llm_path_planner_T_shape', anonymous=True)
        safe_margin = 0.5  # meters
        waypoint_spacing = 1.0  # meters
        environment_data = rospy.get_param('/environment_data_provider/environment_data', {})

        # Example usage (assuming you have current_pose and target_object)
        # current_pose = get_current_pose()  # Replace with actual function to get current pose
        # target_object = get_target_object()  # Replace with actual function to get target object
        # waypoints = generate_waypoints_with_navigation_agent(llm_client, current_pose, target_object, environment_data, safe_margin, waypoint_spacing)

    except rospy.ROSInterruptException:
        pass
