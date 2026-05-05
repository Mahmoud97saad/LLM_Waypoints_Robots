#!/usr/bin/env python3

import rospy
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
import heapq
import time
import subprocess

MAX_SHORT_WAYPOINTS = 5
MIN_LONG_WAYPOINTS = 6
MAX_LONG_WAYPOINTS = 10


# Edit by Shehab: keep path validation slightly farther from remembered obstacle zones to account for
# the robot body footprint without inflating the stored obstacle radius itself.
def _robot_obstacle_safety_margin():
    return float(rospy.get_param('~robot_obstacle_safety_margin', 0.18))


# Edit by Shehab: keep obstacle-detour waypoints off the corridor wall so tracking error does not turn a
# valid geometric path into a wall skim.
def _detour_wall_margin():
    return float(rospy.get_param('~detour_wall_margin', max(0.55, _robot_obstacle_safety_margin() + 0.35)))


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


def calculate_corridor_bounds(environment_data=None, safe_margin=0.5):
    # Extract corridor dimensions from environment_data
    environment_data = environment_data or {}
    main_corridor_width = environment_data.get('main_corridor_width', 18.0)
    corridor_01_length = environment_data.get('corridor_01_length', 29.5)
    corridor_01_width = environment_data.get('corridor_01_width', 5.0)
    corridor_02_length = environment_data.get('corridor_02_length', 29.5)
    corridor_02_width = environment_data.get('corridor_02_width', 5.0)

    return {
        'Main_Corridor': {
            'x_min': -main_corridor_width / 2 + safe_margin,
            'x_max': main_corridor_width / 2 - safe_margin,
            'y_min': -5.0 + safe_margin,
            'y_max': -0.5
        },
        'Corridor_01': {
            'x_min': -corridor_01_width * 2 + 1.5,
            'x_max': -corridor_01_width + safe_margin,
            'y_min': 0.5,
            'y_max': corridor_01_length - safe_margin
        },
        'Corridor_02': {
            'x_min': corridor_02_width - safe_margin,
            'x_max': corridor_02_width * 2 - 1.5,
            'y_min': 0.5,
            'y_max': corridor_02_length - safe_margin
        }
    }


def _point_in_any_bounds(point, corridor_bounds, tolerance=0.0):
    for corridor_name, bounds in corridor_bounds.items():
        if ((bounds['x_min'] - tolerance) <= point['x'] <= (bounds['x_max'] + tolerance) and
                (bounds['y_min'] - tolerance) <= point['y'] <= (bounds['y_max'] + tolerance)):
            return corridor_name
    return None


def _obstacle_radius(obstacle):
    return float(obstacle.get('clearance_radius', obstacle.get('radius', 0.75))) + _robot_obstacle_safety_margin()


# Edit by Shehab: allow a small endpoint tolerance so near-boundary start/end poses are still classified
# as valid corridor motion instead of failing on tiny odom overshoot.
def _segment_stays_in_free_space(start, end, corridor_bounds, step_size=0.05, log_failures=True,
                                 endpoint_tolerance=0.1, endpoint_tolerance_distance=0.2):
    distance = math.hypot(end['x'] - start['x'], end['y'] - start['y'])
    steps = max(1, int(math.ceil(distance / step_size)))

    for step in range(steps + 1):
        ratio = step / float(steps)
        sampled_point = {
            'x': start['x'] + (end['x'] - start['x']) * ratio,
            'y': start['y'] + (end['y'] - start['y']) * ratio
        }
        distance_from_start = distance * ratio
        distance_from_end = distance * (1.0 - ratio)
        tolerance = endpoint_tolerance if (
            distance_from_start <= endpoint_tolerance_distance or
            distance_from_end <= endpoint_tolerance_distance
        ) else 0.0
        if _point_in_any_bounds(sampled_point, corridor_bounds, tolerance=tolerance) is None:
            if log_failures:
                rospy.logwarn(
                    f"Path segment from ({start['x']:.2f}, {start['y']:.2f}) to "
                    f"({end['x']:.2f}, {end['y']:.2f}) leaves free corridor space near "
                    f"({sampled_point['x']:.2f}, {sampled_point['y']:.2f})."
                )
            return False
    return True


def _point_clear_of_detected_obstacles(point, detected_obstacles, log_failures=True):
    for obstacle in detected_obstacles or []:
        radius = _obstacle_radius(obstacle)
        distance = math.hypot(point['x'] - obstacle['x'], point['y'] - obstacle['y'])
        if distance <= radius:
            if log_failures:
                rospy.logwarn(
                    f"Waypoint ({point['x']:.2f}, {point['y']:.2f}) is {distance:.2f} m from "
                    f"detected obstacle ({obstacle['x']:.2f}, {obstacle['y']:.2f}); "
                    f"required clearance is {radius:.2f} m."
                )
            return False
    return True


def _segment_clear_of_detected_obstacles(start, end, detected_obstacles, step_size=0.05, log_failures=True):
    if not detected_obstacles:
        return True

    distance = math.hypot(end['x'] - start['x'], end['y'] - start['y'])
    steps = max(1, int(math.ceil(distance / step_size)))
    start_distances = {
        index: math.hypot(start['x'] - obstacle['x'], start['y'] - obstacle['y'])
        for index, obstacle in enumerate(detected_obstacles)
    }

    for step in range(steps + 1):
        ratio = step / float(steps)
        sampled_point = {
            'x': start['x'] + (end['x'] - start['x']) * ratio,
            'y': start['y'] + (end['y'] - start['y']) * ratio
        }
        for index, obstacle in enumerate(detected_obstacles):
            radius = _obstacle_radius(obstacle)
            obstacle_distance = math.hypot(sampled_point['x'] - obstacle['x'], sampled_point['y'] - obstacle['y'])
            start_distance = start_distances[index]

            if start_distance <= radius:
                if obstacle_distance + 0.02 < start_distance:
                    if log_failures:
                        rospy.logwarn(
                            f"Path segment from ({start['x']:.2f}, {start['y']:.2f}) to "
                            f"({end['x']:.2f}, {end['y']:.2f}) moves deeper into a detected static obstacle zone."
                        )
                    return False
                continue

            if obstacle_distance <= radius:
                if log_failures:
                    rospy.logwarn(
                        f"Path segment from ({start['x']:.2f}, {start['y']:.2f}) to "
                        f"({end['x']:.2f}, {end['y']:.2f}) crosses a detected static obstacle zone."
                    )
                return False
    return True


def _path_makes_target_progress(waypoints, target_position, current_position, tolerance=0.75):
    previous_distance = math.hypot(
        target_position['x'] - current_position['x'],
        target_position['y'] - current_position['y']
    )

    for waypoint in waypoints:
        distance = math.hypot(target_position['x'] - waypoint['x'], target_position['y'] - waypoint['y'])
        if distance > previous_distance + tolerance:
            rospy.logwarn(
                f"Waypoint ({waypoint['x']:.2f}, {waypoint['y']:.2f}) increases target distance "
                f"from {previous_distance:.2f} m to {distance:.2f} m."
            )
            return False
        previous_distance = distance
    return True


def validate_waypoints(waypoints, environment_data, safe_margin, target_position, current_position=None,
                       detected_obstacles=None, enforce_target_progress=True):
    corridors = calculate_corridor_bounds(environment_data, safe_margin)
    corridor_bounds = dict(corridors)
    corridor_bounds['Junction_01'] = {
        'x_min': corridors['Corridor_01']['x_min'],
        'x_max': corridors['Corridor_01']['x_max'],
        'y_min': corridors['Main_Corridor']['y_max'],
        'y_max': corridors['Corridor_01']['y_min']
    }
    corridor_bounds['Junction_02'] = {
        'x_min': corridors['Corridor_02']['x_min'],
        'x_max': corridors['Corridor_02']['x_max'],
        'y_min': corridors['Main_Corridor']['y_max'],
        'y_max': corridors['Corridor_02']['y_min']
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
        if not _point_clear_of_detected_obstacles(waypoint, detected_obstacles):
            return None
        rospy.loginfo(f"Waypoint ({x}, {y}) is within {corridor_name}.")

    if current_position is not None:
        if enforce_target_progress and not _path_makes_target_progress(waypoints, target_position, current_position):
            return None

        segment_points = [current_position] + waypoints
        for start, end in zip(segment_points, segment_points[1:]):
            if not _segment_stays_in_free_space(start, end, corridor_bounds):
                return None
            if not _segment_clear_of_detected_obstacles(start, end, detected_obstacles):
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


def determine_corridor(position, environment_data=None, safe_margin=0.5, boundary_tolerance=0.1):
    corridors = calculate_corridor_bounds(environment_data, safe_margin)

    x = position['x']
    y = position['y']

    # rospy.loginfo(f"Checking if position ({x}, {y}) is within any corridor.")
    for corridor_name, bounds in corridors.items():
        # rospy.loginfo(f"Checking {corridor_name}:")
        # rospy.loginfo(f"  X-axis: {bounds['x_min']} <= {x} <= {bounds['x_max']}")
        # rospy.loginfo(f"  Y-axis: {bounds['y_min']} <= {y} <= {bounds['y_max']}")
        if ((bounds['x_min'] - boundary_tolerance) <= x <= (bounds['x_max'] + boundary_tolerance) and
                (bounds['y_min'] - boundary_tolerance) <= y <= (bounds['y_max'] + boundary_tolerance)):
            rospy.loginfo(f"Position ({x}, {y}) is in {corridor_name}.")
            return corridor_name
    rospy.logwarn(f"Position ({x}, {y}) does not lie within any defined corridor.")
    return None


def _junction_for_corridor(corridor_name, corridors):
    if corridor_name == 'Corridor_01':
        bounds = corridors['Corridor_01']
    elif corridor_name == 'Corridor_02':
        bounds = corridors['Corridor_02']
    else:
        return None

    return {
        'x': (bounds['x_min'] + bounds['x_max']) / 2.0,
        'y': corridors['Main_Corridor']['y_max']
    }


def _same_point(first, second, tolerance=0.01):
    return math.hypot(first['x'] - second['x'], first['y'] - second['y']) <= tolerance


def _append_unique_vertex(vertices, point):
    if not vertices or not _same_point(vertices[-1], point):
        vertices.append({'x': float(point['x']), 'y': float(point['y'])})


def _build_route_vertices(current_position, current_corridor, target_position, target_corridor, corridors):
    vertices = [{'x': float(current_position['x']), 'y': float(current_position['y'])}]

    if current_corridor == target_corridor:
        _append_unique_vertex(vertices, target_position)
        return vertices

    if current_corridor == 'Main_Corridor':
        if target_corridor != 'Main_Corridor':
            target_junction = _junction_for_corridor(target_corridor, corridors)
            if target_junction is None:
                return None
            _append_unique_vertex(vertices, {'x': target_junction['x'], 'y': current_position['y']})
            _append_unique_vertex(vertices, target_junction)
    else:
        current_junction = _junction_for_corridor(current_corridor, corridors)
        if current_junction is None:
            return None
        _append_unique_vertex(vertices, current_junction)

        if target_corridor == 'Main_Corridor':
            _append_unique_vertex(vertices, {'x': target_position['x'], 'y': current_junction['y']})
        else:
            target_junction = _junction_for_corridor(target_corridor, corridors)
            if target_junction is None:
                return None
            _append_unique_vertex(vertices, target_junction)

    _append_unique_vertex(vertices, target_position)
    return vertices


def _corridor_bounds_with_junctions(corridors):
    corridor_bounds = dict(corridors)
    corridor_bounds['Junction_01'] = {
        'x_min': corridors['Corridor_01']['x_min'],
        'x_max': corridors['Corridor_01']['x_max'],
        'y_min': corridors['Main_Corridor']['y_max'],
        'y_max': corridors['Corridor_01']['y_min']
    }
    corridor_bounds['Junction_02'] = {
        'x_min': corridors['Corridor_02']['x_min'],
        'x_max': corridors['Corridor_02']['x_max'],
        'y_min': corridors['Main_Corridor']['y_max'],
        'y_max': corridors['Corridor_02']['y_min']
    }
    return corridor_bounds


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _point_to_segment_distance(point, start, end):
    dx = end['x'] - start['x']
    dy = end['y'] - start['y']
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        return math.hypot(point['x'] - start['x'], point['y'] - start['y'])

    ratio = ((point['x'] - start['x']) * dx + (point['y'] - start['y']) * dy) / length_squared
    ratio = _clamp(ratio, 0.0, 1.0)
    closest = {
        'x': start['x'] + ratio * dx,
        'y': start['y'] + ratio * dy
    }
    return math.hypot(point['x'] - closest['x'], point['y'] - closest['y'])


def _obstacle_intersects_segment(start, end, obstacle):
    radius = _obstacle_radius(obstacle)
    return _point_to_segment_distance(obstacle, start, end) <= radius + 0.10


def _bounds_for_detour(start, end, obstacle, corridor_bounds):
    for point in (obstacle, start, end):
        corridor_name = _point_in_any_bounds(point, corridor_bounds)
        if corridor_name:
            return corridor_bounds[corridor_name]
    return None


def _attempt_clearance_padding(attempt_index):
    # Use a larger obstacle buffer so replanned paths stay materially farther
    # away from detected obstacles instead of skimming the same blocked area.
    return 1.00 + 0.35 * max(0, int(attempt_index) - 1)


def _axis_aligned_obstacle_detour(start, end, obstacle, corridor_bounds, attempt_index=1):
    dx = end['x'] - start['x']
    dy = end['y'] - start['y']
    radius = _obstacle_radius(obstacle)
    clearance = radius + _attempt_clearance_padding(attempt_index)
    bounds = _bounds_for_detour(start, end, obstacle, corridor_bounds)
    if bounds is None:
        return None

    # Keep detours close to the corridor wall and hold that offset for some
    # forward progress before merging back to the nominal route. This avoids a
    # "sidestep then snap back" pattern that can send the robot back toward the
    # same obstacle immediately after the second waypoint.
    wall_margin = _detour_wall_margin()
    forward_progress = max(1.8, clearance * 1.25)

    if abs(dx) <= 0.05:
        direction = 1.0 if dy >= 0.0 else -1.0
        entry_y = obstacle['y'] - direction * clearance
        exit_y = obstacle['y'] + direction * (clearance + forward_progress)
        segment_y_min = min(start['y'], end['y'])
        segment_y_max = max(start['y'], end['y'])
        entry_y = _clamp(
            entry_y,
            max(segment_y_min, bounds['y_min'] + wall_margin),
            min(segment_y_max, bounds['y_max'] - wall_margin)
        )
        exit_y = _clamp(
            exit_y,
            max(segment_y_min, bounds['y_min'] + wall_margin),
            min(segment_y_max, bounds['y_max'] - wall_margin)
        )

        if abs(exit_y - entry_y) < clearance * 0.5:
            return None

        candidates = [
            bounds['x_min'] + wall_margin,
            bounds['x_max'] - wall_margin
        ]
        candidates = [candidate for candidate in candidates if bounds['x_min'] <= candidate <= bounds['x_max']]
        if not candidates:
            return None
        prefer_positive_side = int(attempt_index) % 2 == 1
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (
                0 if (candidate >= obstacle['x']) == prefer_positive_side else 1,
                -abs(candidate - obstacle['x'])
            )
        )
        bypass_x = sorted_candidates[0]
        return [
            {'x': start['x'], 'y': entry_y},
            {'x': bypass_x, 'y': entry_y},
            {'x': bypass_x, 'y': exit_y},
            {'x': end['x'], 'y': exit_y}
        ]

    if abs(dy) <= 0.05:
        direction = 1.0 if dx >= 0.0 else -1.0
        entry_x = obstacle['x'] - direction * clearance
        exit_x = obstacle['x'] + direction * (clearance + forward_progress)
        segment_x_min = min(start['x'], end['x'])
        segment_x_max = max(start['x'], end['x'])
        entry_x = _clamp(
            entry_x,
            max(segment_x_min, bounds['x_min'] + wall_margin),
            min(segment_x_max, bounds['x_max'] - wall_margin)
        )
        exit_x = _clamp(
            exit_x,
            max(segment_x_min, bounds['x_min'] + wall_margin),
            min(segment_x_max, bounds['x_max'] - wall_margin)
        )

        if abs(exit_x - entry_x) < clearance * 0.5:
            return None

        candidates = [
            bounds['y_min'] + wall_margin,
            bounds['y_max'] - wall_margin
        ]
        candidates = [candidate for candidate in candidates if bounds['y_min'] <= candidate <= bounds['y_max']]
        if not candidates:
            return None
        prefer_positive_side = int(attempt_index) % 2 == 1
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (
                0 if (candidate >= obstacle['y']) == prefer_positive_side else 1,
                -abs(candidate - obstacle['y'])
            )
        )
        bypass_y = sorted_candidates[0]
        return [
            {'x': entry_x, 'y': start['y']},
            {'x': entry_x, 'y': bypass_y},
            {'x': exit_x, 'y': bypass_y},
            {'x': exit_x, 'y': end['y']}
        ]

    return None


def _route_with_obstacle_detours(route_vertices, detected_obstacles, corridors, attempt_index=1):
    if not detected_obstacles:
        return route_vertices

    corridor_bounds = _corridor_bounds_with_junctions(corridors)
    safe_vertices = [route_vertices[0]]

    for start, end in zip(route_vertices, route_vertices[1:]):
        direct_segment_clear = (
            _segment_stays_in_free_space(start, end, corridor_bounds, log_failures=False) and
            _segment_clear_of_detected_obstacles(start, end, detected_obstacles, log_failures=False)
        )
        if direct_segment_clear:
            _append_unique_vertex(safe_vertices, end)
            continue

        segment_vertices = None
        start_inside_obstacle = any(
            math.hypot(start['x'] - obstacle['x'], start['y'] - obstacle['y']) <= _obstacle_radius(obstacle)
            for obstacle in detected_obstacles
        )

        # Edit by Shehab: when replanning starts from inside the remembered obstacle zone, prefer an
        # axis-aligned egress path over visibility sampling so the first segment moves out of the blocked
        # area instead of skimming along its edge.
        if start_inside_obstacle:
            for obstacle in detected_obstacles:
                if not _obstacle_intersects_segment(start, end, obstacle):
                    continue
                for extra_attempt in range(3):
                    segment_vertices = _axis_aligned_obstacle_detour(
                        start,
                        end,
                        obstacle,
                        corridor_bounds,
                        attempt_index=attempt_index + extra_attempt
                    )
                    if segment_vertices:
                        rospy.loginfo(
                            f"Inserted axis-aligned obstacle detour around "
                            f"({obstacle['x']:.2f}, {obstacle['y']:.2f})."
                        )
                        break
                if segment_vertices:
                    break

        if not segment_vertices:
            segment_vertices = _find_visibility_detour(
                start, end, detected_obstacles, corridor_bounds, attempt_index=attempt_index
            )

        if not segment_vertices:
            for obstacle in detected_obstacles:
                if not _obstacle_intersects_segment(start, end, obstacle):
                    continue
                segment_vertices = _axis_aligned_obstacle_detour(
                    start, end, obstacle, corridor_bounds, attempt_index=attempt_index
                )
                if segment_vertices:
                    rospy.loginfo(
                        f"Inserted axis-aligned obstacle detour around "
                        f"({obstacle['x']:.2f}, {obstacle['y']:.2f})."
                    )
                    break

        if segment_vertices:
            rospy.loginfo(
                f"Inserted multi-obstacle detour between ({start['x']:.2f}, {start['y']:.2f}) "
                f"and ({end['x']:.2f}, {end['y']:.2f})."
            )
            for vertex in segment_vertices:
                _append_unique_vertex(safe_vertices, vertex)
        _append_unique_vertex(safe_vertices, end)

    return safe_vertices


def _find_visibility_detour(start, end, detected_obstacles, corridor_bounds, attempt_index=1):
    nodes = [
        {'x': float(start['x']), 'y': float(start['y'])},
        {'x': float(end['x']), 'y': float(end['y'])}
    ]

    for obstacle in detected_obstacles:
        bounds = _bounds_for_detour(start, end, obstacle, corridor_bounds)
        if not bounds:
            continue

        radius = _obstacle_radius(obstacle)
        clearance = radius + _attempt_clearance_padding(attempt_index)
        candidate_points = [
            {'x': obstacle['x'] - clearance, 'y': obstacle['y']},
            {'x': obstacle['x'] + clearance, 'y': obstacle['y']},
            {'x': obstacle['x'], 'y': obstacle['y'] - clearance},
            {'x': obstacle['x'], 'y': obstacle['y'] + clearance},
            {'x': obstacle['x'] - clearance, 'y': obstacle['y'] - clearance},
            {'x': obstacle['x'] + clearance, 'y': obstacle['y'] + clearance},
            {'x': obstacle['x'] - clearance, 'y': obstacle['y'] + clearance},
            {'x': obstacle['x'] + clearance, 'y': obstacle['y'] - clearance}
        ]

        wall_margin = _detour_wall_margin()
        wall_offset = wall_margin + (0.08 * (max(0, int(attempt_index) - 1) % 3))
        candidate_points.extend([
            {'x': bounds['x_min'] + wall_offset, 'y': obstacle['y'] - clearance},
            {'x': bounds['x_min'] + wall_offset, 'y': obstacle['y'] + clearance},
            {'x': bounds['x_max'] - wall_offset, 'y': obstacle['y'] - clearance},
            {'x': bounds['x_max'] - wall_offset, 'y': obstacle['y'] + clearance}
        ])

        for point in candidate_points:
            point = {
                'x': _clamp(point['x'], bounds['x_min'] + wall_margin, bounds['x_max'] - wall_margin),
                'y': _clamp(point['y'], bounds['y_min'] + wall_margin, bounds['y_max'] - wall_margin)
            }
            if _point_in_any_bounds(point, corridor_bounds) is None:
                continue
            if not _point_clear_of_detected_obstacles(point, detected_obstacles, log_failures=False):
                continue
            if any(_same_point(point, existing, tolerance=0.05) for existing in nodes):
                continue
            nodes.append(point)

    graph = {index: [] for index in range(len(nodes))}
    for first_index, first in enumerate(nodes):
        for second_index in range(first_index + 1, len(nodes)):
            second = nodes[second_index]
            if not _segment_stays_in_free_space(first, second, corridor_bounds, log_failures=False):
                continue
            if not _segment_clear_of_detected_obstacles(first, second, detected_obstacles, log_failures=False):
                continue
            length = math.hypot(second['x'] - first['x'], second['y'] - first['y'])
            graph[first_index].append((second_index, length))
            graph[second_index].append((first_index, length))

    path_indices = _shortest_path_indices(graph, 0, 1, nodes)
    if not path_indices or len(path_indices) <= 2:
        return None

    return [nodes[index] for index in path_indices[1:-1]]


def _shortest_path_indices(graph, start_index, end_index, nodes):
    queue = [(0.0, 0.0, start_index)]
    distances = {start_index: 0.0}
    previous = {}

    while queue:
        _, current_cost, current_index = heapq.heappop(queue)
        if current_index == end_index:
            break
        if current_cost > distances.get(current_index, float('inf')):
            continue

        for neighbor_index, edge_cost in graph[current_index]:
            new_cost = current_cost + edge_cost
            if new_cost >= distances.get(neighbor_index, float('inf')):
                continue
            distances[neighbor_index] = new_cost
            previous[neighbor_index] = current_index
            heuristic = math.hypot(
                nodes[end_index]['x'] - nodes[neighbor_index]['x'],
                nodes[end_index]['y'] - nodes[neighbor_index]['y']
            )
            heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor_index))

    if end_index not in distances:
        return None

    path = [end_index]
    while path[-1] != start_index:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def _distribute_extra_points(segment_lengths, extra_point_count):
    if extra_point_count <= 0 or not segment_lengths:
        return [0 for _ in segment_lengths]

    total_length = sum(segment_lengths)
    if total_length <= 0.0:
        return [0 for _ in segment_lengths]

    allocations = []
    remainders = []
    allocated = 0
    for index, length in enumerate(segment_lengths):
        exact = extra_point_count * (length / total_length)
        base = int(math.floor(exact))
        allocations.append(base)
        remainders.append((exact - base, index))
        allocated += base

    for _, index in sorted(remainders, reverse=True)[:extra_point_count - allocated]:
        allocations[index] += 1

    return allocations


# Edit by Shehab: do not insert extra interpolated points on segments that start inside a remembered
# obstacle zone; those synthetic points can drag the first escape path back toward the same obstacle.
def _sample_route_vertices(route_vertices, waypoint_count, detected_obstacles=None):
    if len(route_vertices) < 2:
        return []

    required_endpoints = len(route_vertices) - 1
    waypoint_count = max(required_endpoints, int(waypoint_count))

    segments = list(zip(route_vertices, route_vertices[1:]))
    segment_lengths = [
        math.hypot(end['x'] - start['x'], end['y'] - start['y'])
        for start, end in segments
    ]
    blocked_segment_indices = set()
    for index, (start, end) in enumerate(segments):
        if any(
            math.hypot(start['x'] - obstacle['x'], start['y'] - obstacle['y']) <= _obstacle_radius(obstacle)
            for obstacle in (detected_obstacles or [])
        ):
            blocked_segment_indices.add(index)
    allocatable_segment_lengths = [
        length if (length >= 1.5 and index not in blocked_segment_indices) else 0.0
        for index, length in enumerate(segment_lengths)
    ]
    extra_allocations = _distribute_extra_points(
        allocatable_segment_lengths,
        waypoint_count - required_endpoints
    )

    waypoints = []
    for (start, end), extra_count in zip(segments, extra_allocations):
        for index in range(1, extra_count + 1):
            ratio = index / float(extra_count + 1)
            waypoints.append({
                'x': start['x'] + (end['x'] - start['x']) * ratio,
                'y': start['y'] + (end['y'] - start['y']) * ratio
            })
        waypoints.append({'x': end['x'], 'y': end['y']})

    return waypoints


def generate_deterministic_waypoints(current_position, current_corridor, target_position, target_corridor,
                                     corridors, waypoint_count, detected_obstacles=None, attempt_index=1):
    route_vertices = _build_route_vertices(
        current_position,
        current_corridor,
        target_position,
        target_corridor,
        corridors
    )
    if not route_vertices:
        return None

    route_vertices = _route_with_obstacle_detours(
        route_vertices,
        detected_obstacles or [],
        corridors,
        attempt_index=attempt_index
    )
    waypoints = _sample_route_vertices(route_vertices, waypoint_count, detected_obstacles=detected_obstacles)
    for waypoint in waypoints:
        waypoint['x'] = round(waypoint['x'], 2)
        waypoint['y'] = round(waypoint['y'], 2)
    return waypoints


def plot_waypoints(current_position, target_position, waypoints, environment_data, safe_margin, junction_point=None,
                   detected_obstacles=None, attempt_label=None, status_label=None):
    corridors = calculate_corridor_bounds(environment_data, safe_margin)
    main_corridor_bounds = corridors['Main_Corridor']
    corridor_01_bounds = corridors['Corridor_01']
    corridor_02_bounds = corridors['Corridor_02']

    rospy.loginfo("Plotting waypoints and corridor boundaries.")

    # Set fixed figure size (e.g., 10 inches by 8 inches)
    fig, ax = plt.subplots(figsize=(7, 6))  # Adjust the size as needed

    # Draw Main Corridor
    main_corridor = plt.Rectangle((main_corridor_bounds['x_min'], main_corridor_bounds['y_min']),
                                  main_corridor_bounds['x_max'] - main_corridor_bounds['x_min'],
                                  main_corridor_bounds['y_max'] - main_corridor_bounds['y_min'],
                                  edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Main Corridor')
    ax.add_patch(main_corridor)

    # Draw Corridor_01
    corridor_01 = plt.Rectangle((corridor_01_bounds['x_min'], corridor_01_bounds['y_min']),
                                corridor_01_bounds['x_max'] - corridor_01_bounds['x_min'],
                                corridor_01_bounds['y_max'] - corridor_01_bounds['y_min'],
                                edgecolor='green', facecolor='lightgreen', alpha=0.5, label='Corridor_01')
    ax.add_patch(corridor_01)

    # Draw Corridor_02
    corridor_02 = plt.Rectangle((corridor_02_bounds['x_min'], corridor_02_bounds['y_min']),
                                corridor_02_bounds['x_max'] - corridor_02_bounds['x_min'],
                                corridor_02_bounds['y_max'] - corridor_02_bounds['y_min'],
                                edgecolor='purple', facecolor='thistle', alpha=0.5, label='Corridor_02')
    ax.add_patch(corridor_02)

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
        #ax.plot(junction_point[0], junction_point[2], 'mo', label='Junction Point')

    for obstacle in detected_obstacles or []:
        radius = float(obstacle.get('radius', 0.75))
        obstacle_circle = plt.Circle(
            (obstacle['x'], obstacle['y']),
            radius,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            linewidth=1.5,
            label='Detected Obstacle'
        )
        ax.add_patch(obstacle_circle)
        ax.plot(obstacle['x'], obstacle['y'], 'rx')

    if attempt_label or status_label:
        title_parts = [part for part in (attempt_label, status_label) if part]
        ax.set_title(" - ".join(title_parts))

    ax.set_xlabel('X-axis (meters)', fontweight='bold')
    ax.set_ylabel('Y-axis (meters)', fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    plt.axis('equal')
    save_dir = (
        rospy.get_param('~path_plot_dir', '/home/shehab/Pictures/path_attempts')
        if hasattr(rospy, 'get_param') else
        '/home/shehab/Pictures/path_attempts'
    )
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError as e:
        fallback_dir = '/tmp/path_attempts'
        rospy.logwarn(f"Could not create path plot directory {save_dir}: {e}. Falling back to {fallback_dir}.")
        save_dir = fallback_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    millis = int((time.time() % 1) * 1000)
    safe_attempt_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', attempt_label or 'path')
    safe_status_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', status_label or 'candidate')
    filename = os.path.join(save_dir, f'{timestamp}_{millis:03d}_{safe_attempt_label}_{safe_status_label}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    latest_filename = os.path.join(os.path.dirname(save_dir), 'E1.png')
    try:
        plt.savefig(latest_filename, dpi=300, bbox_inches='tight')
    except OSError as e:
        rospy.logwarn(f"Could not update latest path figure {latest_filename}: {e}")
    plt.close(fig)
    if rospy.get_param('~popup_path_plots', False):
        display_value = os.environ.get('DISPLAY')
        if display_value:
            try:
                subprocess.Popen(['xdg-open', filename])
            except Exception as e:
                rospy.logwarn(f"Could not open path figure popup for {filename}: {e}")
        else:
            rospy.logwarn("popup_path_plots is enabled but DISPLAY is not set; skipping popup.")
    rospy.loginfo(f"Saved path figure to {filename}")
    return filename


def generate_waypoints_with_navigation_agent(llm_client, current_pose, target_object, environment_data, safe_margin,
                                             waypoint_spacing, detected_obstacles=None, dynamic_obstacles=None,
                                             attempt_index=1, planning_cycle_index=1, planning_mode='initial'):
    max_attempts = rospy.get_param('~max_attempts', 5)  # Maximum attempts per agent

    # Calculate junction points
    junction_point_1 = (-6.5, -0.5)
    junction_point_2 = (6.5, -0.5)

    corridors = calculate_corridor_bounds(environment_data, safe_margin)
    main_corridor_bounds = corridors['Main_Corridor']
    corridor_01_bounds = corridors['Corridor_01']
    corridor_02_bounds = corridors['Corridor_02']

    # Determine robot's current corridor
    current_position = {
        'x': current_pose.pose.position.x,
        'y': current_pose.pose.position.y
    }
    rospy.loginfo(f"Determining current corridor for position ({current_position['x']}, {current_position['y']}).")
    current_corridor = determine_corridor(current_position, environment_data, safe_margin)

    if current_corridor is None:
        rospy.logerr("Cannot determine the current corridor of the robot.")
        return None

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
    detected_obstacles = detected_obstacles if detected_obstacles is not None else dynamic_obstacles
    detected_obstacles = detected_obstacles or []
    if detected_obstacles:
        rospy.loginfo(f"Planning with {len(detected_obstacles)} detected static obstacle(s).")

    target_corridor = determine_corridor(target_position, environment_data, safe_margin)

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
                if target_corridor == 'Corridor_01':
                    junction_point = junction_point_1
                elif target_corridor == 'Corridor_02':
                    junction_point = junction_point_2
                else:
                    rospy.logwarn(f"No junction point defined for target corridor: {target_corridor}.")
                    junction_point = None
            else:
                # For non-Main Corridors, assign junction point based on current corridor
                if current_corridor == 'Corridor_01':
                    junction_point = junction_point_1
                elif current_corridor == 'Corridor_02':
                    junction_point = junction_point_2
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

    if target_corridor is not None:
        deterministic_waypoints = generate_deterministic_waypoints(
            current_position,
            current_corridor,
            target_position,
            target_corridor,
            corridors,
            waypoint_count,
            detected_obstacles,
            attempt_index=attempt_index
        )
        if deterministic_waypoints:
            rospy.loginfo("Generated deterministic corridor waypoints; validating before publish.")
            try:
                plot_waypoints(
                    current_position,
                    target_position,
                    deterministic_waypoints,
                    environment_data,
                    safe_margin,
                    junction_point,
                    detected_obstacles,
                    f'{planning_mode}_cycle_{planning_cycle_index}_attempt_{attempt_index}_deterministic',
                    'candidate'
                )
            except Exception as e:
                rospy.logerr(f"Failed to plot deterministic candidate waypoints: {e}")
            validated_waypoints = validate_waypoints(
                deterministic_waypoints,
                environment_data,
                safe_margin,
                target_object['position'],
                current_position,
                detected_obstacles,
                enforce_target_progress=(current_corridor == target_corridor)
            )
            if validated_waypoints:
                rospy.loginfo("Deterministic corridor planner succeeded.")
                try:
                    plot_waypoints(
                        current_position,
                        target_position,
                        validated_waypoints,
                        environment_data,
                        safe_margin,
                        junction_point,
                        detected_obstacles,
                        f'{planning_mode}_cycle_{planning_cycle_index}_attempt_{attempt_index}_deterministic',
                        'validated'
                    )
                except Exception as e:
                    rospy.logerr(f"Failed to plot waypoints: {e}")
                return validated_waypoints
            rospy.logwarn("Deterministic corridor planner failed validation; falling back to LLM planner.")

    # Construct system and user prompts
    obstacle_prompt = "No static obstacle has been detected on the current route yet."
    if detected_obstacles:
        obstacle_lines = [
            f"- center ({obstacle['x']:.2f}, {obstacle['y']:.2f}), clearance radius {obstacle.get('radius', 0.75):.2f} m"
            for obstacle in detected_obstacles
        ]
        obstacle_prompt = "Detected static obstacle zones to avoid:\n" + "\n".join(obstacle_lines)

    navigation_system_prompt = f"""
Robot's current position: ({current_position['x']:.2f}, {current_position['y']:.2f})
Target position: ({target_x:.2f}, {target_y:.2f})
Corridor boundaries:

Main Corridor:
- X-axis: from {main_corridor_bounds['x_min']:.2f} to {main_corridor_bounds['x_max']:.2f}
- Y-axis: from {main_corridor_bounds['y_min']:.2f} to {main_corridor_bounds['y_max']:.2f}

Corridor_01:
- X-axis: from {corridor_01_bounds['x_min']:.2f} to {corridor_01_bounds['x_max']:.2f}
- Y-axis: from {corridor_01_bounds['y_min']:.2f} to {corridor_01_bounds['y_max']:.2f}

Corridor_02:
- X-axis: from {corridor_02_bounds['x_min']:.2f} to {corridor_02_bounds['x_max']:.2f}
- Y-axis: from {corridor_02_bounds['y_min']:.2f} to {corridor_02_bounds['y_max']:.2f}

Safe margin: {safe_margin} meters from walls
Robot-to-target distance: {robot_to_target_distance:.2f} meters
Waypoint spacing target: {waypoint_spacing} meters
Required waypoint count: {waypoint_count}
Planning mode: {planning_mode}
Planning cycle: {planning_cycle_index}
Retry attempt: {attempt_index} / {max_attempts}
{obstacle_prompt}

You are a navigation assistant for a mobile robot in a U-shaped corridor map with three corridors: Main Corridor, Corridor_01, and Corridor_02.
Provide a concise sequence of waypoints as (x, y) coordinates for the robot to follow to reach the destination, avoiding obstacles and maintaining the safe margins defined above. Ensure that:
1. All waypoints are within the corridor boundaries defined above.
2. Help in creating a list of waypoints from the robot's current position to the target position in the best sequence on both axis.
3. Room_number_plates from 101 to 114 are in Corridor_01, Room_number_plates 201 to 218 and windows are in Corridor_02, and other objects like Stairs are in the Main Corridor.
4. If the robot needs to move from one corridor to another, include the appropriate junction point ({junction_point_str}) in the waypoints as the first waypoint and the next waypoint must be a junction point ({junction_point_str}) of the corridor where the target is located.
5. Ensure the path is direct and efficient, facilitating smooth transitions between corridors when necessary.
6. Generate exactly {waypoint_count} waypoint(s), based on the robot-to-target distance and waypoint spacing. Short routes must use fewer than 6 waypoints; long routes must use between 6 and 10 waypoints, never more than 10.
7. Do not place any waypoint inside a detected static obstacle zone, and do not make any path segment pass through a detected static obstacle zone.
8. Each waypoint must reduce or maintain distance to the target compared with the previous robot/waypoint position; do not backtrack away from the target.
9. Do not include any comments, annotations, or additional text. The output should be a valid JSON array of waypoints only.
10. Format the output strictly as a JSON array of coordinates without any comments or additional text.
11. Please follow all rules; it's a humble request.
"""

    navigation_user_prompt = f"""

Robot's current position: ({current_position['x']:.2f}, {current_position['y']:.2f})
Target position: ({target_x:.2f}, {target_y:.2f})

Safe margin: {safe_margin} meters from walls
Robot-to-target distance: {robot_to_target_distance:.2f} meters
Waypoint spacing target: {waypoint_spacing} meters
Required waypoint count: {waypoint_count}
Planning mode: {planning_mode}
Planning cycle: {planning_cycle_index}
Retry attempt: {attempt_index} / {max_attempts}
{obstacle_prompt}

Please provide a concise sequence of waypoints as (x, y) coordinates for the robot to follow to reach the destination, avoiding obstacles and maintaining
the safe margins defined above. Ensure that:
1. The waypoints form a continuous and logical path towards the destination that reduces the distance between the robot's current position and the target object without any backtracking.
2. Generate exactly {waypoint_count} waypoint(s), including the final target waypoint. This count is computed from the robot-to-target distance; short routes must produce fewer than 6 waypoints, while long routes must produce between 6 and 10 waypoints and never more than 10.
3. **All waypoints must form a straight or smoothly curved path towards the target without deviating away**.
4. The final waypoint must be exactly at the target object's position.
5. Do not include the current position in the waypoints.
6. Each waypoint should be about {waypoint_spacing} meters apart from the previous one to avoid redundancy; the final target waypoint may be closer if needed to end exactly at the target.
7. Ensure that the X-axis and Y-axis values of all waypoints remain within the corridor's boundaries without deviation.
8. {'If the robot needs to move from one corridor to another, include the appropriate junction point (' + junction_point_str + ') in the waypoints as the first waypoint; otherwise, generate waypoints normally as you are generating.' if junction_point else 'Generate waypoints directly towards the target without using any junction points.'}
9. Avoid every detected static obstacle zone listed above. No waypoint or straight path segment between waypoints may enter those radius zones.
10. Do not backtrack away from the target; every waypoint must reduce or maintain the remaining target distance.
11. Please follow all rules; it's a humble request.

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
        varied_user_prompt = navigation_user_prompt + (
            f"\nRetry variation hint: current outer retry is {attempt_index}; "
            f"inner LLM try is {attempt + 1}. If the previous try failed, choose a distinct valid path shape.\n"
        )
        navigation_output = agents(llm_client, navigation_system_prompt, varied_user_prompt)
        if navigation_output:
            waypoints = parse_waypoints(navigation_output)
            if waypoints:
                waypoints = enforce_waypoint_count(waypoints, waypoint_count)
                try:
                    plot_waypoints(
                        current_position,
                        target_position,
                        waypoints,
                        environment_data,
                        safe_margin,
                        junction_point,
                        detected_obstacles,
                        f'{planning_mode}_cycle_{planning_cycle_index}_attempt_{attempt_index}_llm_try_{attempt + 1}',
                        'candidate'
                    )
                except Exception as e:
                    rospy.logerr(f"Failed to plot LLM candidate waypoints: {e}")
                # Validate waypoints here
                waypoints = validate_waypoints(
                    waypoints,
                    environment_data,
                    safe_margin,
                    target_object['position'],
                    current_position,
                    detected_obstacles
                )
                if waypoints:
                    rospy.loginfo(f"Navigation Agent succeeded on attempt {attempt + 1}")
                    # Optional: Visualize waypoints
                    try:
                        plot_waypoints(
                            current_position,
                            target_position,
                            waypoints,
                            environment_data,
                            safe_margin,
                            junction_point,
                            detected_obstacles,
                            f'{planning_mode}_cycle_{planning_cycle_index}_attempt_{attempt_index}_llm_try_{attempt + 1}',
                            'validated'
                        )
                    except Exception as e:
                        rospy.logerr(f"Failed to plot waypoints: {e}")
                    return waypoints
        rospy.logwarn(f"Navigation Agent attempt {attempt + 1} failed. Retrying...")
    else:
        rospy.logerr("Navigation Agent failed after maximum attempts.")
        return None


if __name__ == '__main__':
    try:
        # Initialize ROS node
        rospy.init_node('llm_path_planner_u_shape', anonymous=True)
        safe_margin = 0.5  # meters
        waypoint_spacing = 1.0  # meters
        environment_data = rospy.get_param('/environment_data_provider/environment_data', {})

    except rospy.ROSInterruptException:
        pass
