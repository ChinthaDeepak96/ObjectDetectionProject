import numpy as np
import cv2
import math
from steering_angle import calculate_steering_angle, display_steering_angle


def calculate_steering_angle(frame, lines):
    height, width, _ = frame.shape
    if lines is None:
        return 0  # no lines → go straight

    left_lines = []
    right_lines = []

    # Separate lines by slope
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 0.0001)

        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))

    # No valid lines
    if not left_lines or not right_lines:
        return 0

    # Average left lane
    left_avg = np.mean(left_lines, axis=0).astype(int)
    x1_l, y1_l, x2_l, y2_l = left_avg

    # Average right lane
    right_avg = np.mean(right_lines, axis=0).astype(int)
    x1_r, y1_r, x2_r, y2_r = right_avg

    # Lane center
    lane_center_x = int((x2_l + x2_r) / 2)

    # Car center
    car_center_x = width // 2

    # Steering offset
    dx = lane_center_x - car_center_x
    dy = int(height * 0.6)

    angle_radians = math.atan(dx / dy)
    angle_degrees = math.degrees(angle_radians)

    return int(angle_degrees)


def display_steering_angle(frame, angle):
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height

    steering_x = int(center_x + angle * 2)
    steering_y = int(center_y - 100)

    cv2.line(frame, (center_x, center_y), (steering_x, steering_y), (0, 0, 255), 5)
    cv2.putText(frame, f"Angle: {angle} deg", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return frame
