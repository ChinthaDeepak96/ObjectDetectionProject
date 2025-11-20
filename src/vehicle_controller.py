class VehicleController:
    def __init__(self):
        self.state = "DRIVE"
        self.speed = 1.0
        self.last_angle = 0
        self.avoid_direction = None  # "LEFT" or "RIGHT"

    def update(self, angle, obstacle_detected, lane_available, obstacle_pos=None):
        """
        angle: steering angle predicted from lane
        obstacle_detected: True/False
        lane_available: True/False
        obstacle_pos: "LEFT", "RIGHT", "CENTER"
        """

        # --------------------------------------------------
        # A) Full obstruction case (no lane + obstacle)
        # --------------------------------------------------
        if obstacle_detected and not lane_available:
            self.state = "STOP"
            self.speed = 0
            return self.state, self.speed

        # --------------------------------------------------
        # B) Obstacle Avoidance Logic
        # --------------------------------------------------
        if obstacle_detected:
            # Determine avoidance direction
            if obstacle_pos == "CENTER":
                # Prefer left lane
                self.state = "AVOID_LEFT"
                self.avoid_direction = "LEFT"
                self.speed = 0.5
                self.last_angle = -25  # turn left

            elif obstacle_pos == "LEFT":
                self.state = "AVOID_RIGHT"
                self.avoid_direction = "RIGHT"
                self.speed = 0.5
                self.last_angle = 25  # turn right

            elif obstacle_pos == "RIGHT":
                self.state = "AVOID_LEFT"
                self.avoid_direction = "LEFT"
                self.speed = 0.5
                self.last_angle = -25

            return self.state, self.speed

        # --------------------------------------------------
        # C) Recenter after avoiding
        # --------------------------------------------------
        if self.avoid_direction is not None:
            # Return to center slowly
            self.state = "RECENTER"
            self.speed = 0.7
            self.last_angle = angle * 0.5  # soften the return
            self.avoid_direction = None
            return self.state, self.speed

        # --------------------------------------------------
        # D) Normal driving behavior
        # --------------------------------------------------
        if not lane_available:
            self.state = "SLOW"
            self.speed = 0.3
            return self.state, self.speed

        if angle > 15:
            self.state = "TURN_RIGHT"
            self.speed = 0.7
        elif angle < -15:
            self.state = "TURN_LEFT"
            self.speed = 0.7
        else:
            self.state = "DRIVE"
            self.speed = 1.0

        self.last_angle = angle
        return self.state, self.speed

    def get_commands(self):
        return self.last_angle, self.speed
