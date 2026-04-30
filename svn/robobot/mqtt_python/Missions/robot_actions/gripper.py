"""Servo gripper actions: high-level interface for gripper servo control.

Gripper configuration values are read from robot.ini under [servogrip].
To tune the gripper, only change values in robot.ini:
    [servogrip]
    open_position  = -950
    mid_position   = -500
    close_position = -300 
    velocity       = 200
    servo_idx      = 2
"""

import configparser


class ServoGripActions:
    """High-level interface for gripper servo control.

    Wraps the servo send command to provide a clean, objective-level interface
    for controlling the gripper servo. Other modules can use this class
    to trigger gripper movements without needing to know servo positions or indices.

    All configuration is read from robot.ini under [servogrip].
    """

    def __init__(self, service, config_path=None):
        self.service = service

        if config_path is None:
            import os
            config_path = os.path.join(
                os.path.dirname(__file__),
                "../../../teensy_interface/robot.ini",
            )

        config = configparser.ConfigParser()
        config.read(config_path)

        if not config.has_section("servogrip"):
            print(f"[ServoGrip] Warning: [servogrip] section missing from {config_path}, using defaults")

        self.servo_idx = config.getint("servogrip", "servo_idx", fallback=2)
        self.grip_open_pos = config.getint("servogrip", "open_position", fallback=-700)
        self.grip_mid_pos = config.getint("servogrip", "mid_position", fallback=-500)
        self.grip_close_pos = config.getint("servogrip", "close_position", fallback=-300)
        self.grip_velocity = config.getint("servogrip", "velocity", fallback=200)

        print(f"[ServoGrip] Setup complete")
        print(f"[ServoGrip] open_pos={self.grip_open_pos}")
        print(f"[ServoGrip] mid_pos={self.grip_mid_pos}")
        print(f"[ServoGrip] close_pos={self.grip_close_pos}")
        print(f"[ServoGrip] velocity={self.grip_velocity}")
        print(f"[ServoGrip] servo_idx={self.servo_idx}")

    def move_to(self, position, speed=None):
        speed_value = speed if speed is not None else self.grip_velocity
        self.service.send("robobot/cmd/T0", f"servo {self.servo_idx} {int(position)} {int(speed_value)}")

    def open(self, speed=None):
        self.move_to(self.grip_open_pos, speed=speed)

    def middle(self, speed=None):
        self.move_to(self.grip_mid_pos, speed=speed)

    def close(self, speed=None):
        self.move_to(self.grip_close_pos, speed=speed)