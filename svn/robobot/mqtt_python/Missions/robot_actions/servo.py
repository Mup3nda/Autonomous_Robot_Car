"""Servo arm actions: high-level interface for arm servo control.

Arm configuration values are read from robot.ini under [servoarm].
To tune the arm, only change values in robot.ini:
    [servoarm]
    up_position   = -900
    mid_position  = -250
    down_position = 120
    velocity      = 200
    servo_idx     = 1
"""

import configparser


class ServoArmActions:
    """High-level interface for arm servo control.
    
    Wraps the servo send command to provide a clean, objective-level interface
    for controlling the 1DOF arm servo. Other modules can use this class
    to trigger arm movements without needing to know servo positions or indices.
    
    All configuration is read from robot.ini under [servoarm].
    
    Args:
        service: Instance of uservice.UService (the low-level service interface)
        config_path: Path to robot.ini (default = "robot.ini")
    
    Example usage:
        arm = ServoArmActions(service)
        arm.move_up()    # arm goes to upright position
        arm.move_mid()   # arm goes to mid position
        arm.move_down()  # arm goes to deployed/down position
    """

    def __init__(self, service, config_path=None):
        self.service = service
        self.memory = None

        if config_path is None:
            import os
            # Fix path: robot_actions -> Missions -> mqtt_python -> robobot -> teensy_interface
            config_path = os.path.join(os.path.dirname(__file__), 
                          "../../../teensy_interface/robot.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        if not config.has_section("servoarm"):
            print(f"[ServoArm] Warning: [servoarm] section missing from {config_path}, using defaults")

        self.servo_idx    = config.getint("servoarm", "servo_idx",     fallback=1)
        self.arm_up_pos   = config.getint("servoarm", "up_position",   fallback=-900)
        self.arm_mid_pos  = config.getint("servoarm", "mid_position",  fallback=-250)
        self.arm_down_pos = config.getint("servoarm", "down_position", fallback=120)
        self.arm_velocity = config.getint("servoarm", "velocity",      fallback=200)

        print(f"[ServoArm] Setup complete")
        print(f"[ServoArm] up_pos={self.arm_up_pos}")
        print(f"[ServoArm] mid_pos={self.arm_mid_pos}")
        print(f"[ServoArm] down_pos={self.arm_down_pos}")
        print(f"[ServoArm] velocity={self.arm_velocity}")
        print(f"[ServoArm] servo_idx={self.servo_idx}")

    def bind_memory(self, memory):
        """Bind mission memory so arm commands can persist desired state."""
        self.memory = memory

    def _store_state(self, target_pos, speed):
        if self.memory is None:
            return
        self.memory["_arm_state"] = {
            "target_pos": int(target_pos),
            "speed": int(speed),
        }

    def move_to(self, position, speed=None, update_memory=True):
        """Move arm to an explicit servo position."""
        s = speed if speed is not None else self.arm_velocity
        self.service.send("robobot/cmd/T0", f"servo {self.servo_idx} {int(position)} {int(s)}")
        if update_memory:
            self._store_state(position, s)

    def hold_position(self, speed=None):
        """Re-send the last commanded target position without changing memory state."""
        if self.memory is None:
            return
        state = self.memory.get("_arm_state")
        if not state:
            return
        hold_speed = speed if speed is not None else state.get("speed", self.arm_velocity)
        self.move_to(state.get("target_pos", self.arm_mid_pos), speed=hold_speed, update_memory=False)

    def move_up(self, speed=None, update_memory=True):
        """Move arm to full upright/resting position.
        
        Call this when the arm is not needed or when no ball is detected.
        
        Args:
            speed: Movement speed (optional, defaults to value in robot.ini)
        """
        self.move_to(self.arm_up_pos, speed=speed, update_memory=update_memory)

    def move_mid(self, speed=None, update_memory=True):
        """Move arm to mid position.
            
        Call this to move arm to a mid position which is responsible to push the box's down.
        Position configured in robot.ini (mid_position, default -250).

        Args:
            speed: Movement speed (optional, defaults to value in robot.ini)
        """
        self.move_to(self.arm_mid_pos, speed=speed, update_memory=update_memory)

    def move_down(self, speed=None, update_memory=True):
        """Move arm to full down/deployed position.
        
        Call this when ball is detected and arm needs to deploy.
        
        Args:
            speed: Movement speed (optional, defaults to value in robot.ini)
        """
        self.move_to(self.arm_down_pos, speed=speed, update_memory=update_memory)