/*  
 * cservoarm.cpp
 * 
 * High-level interface for arm servo control.
 * Other modules (e.g. ball detection) can include this and call
 * moveUp() / moveDown() directly.
 * 
 * Arm configuration values are read from robot.ini under [servoarm].
 * To tune the arm, only change values in robot.ini:
 *     [servoarm]
 *     up_position   = -900
 *     mid_position  = -250
 *     down_position = 120
 *     velocity      = 200
 *     servo_idx     = 1
 * 
 * Copyright © 2023 DTU, Christian Andersen jcan@dtu.dk
 * The MIT License (MIT)  https://mit-license.org/
 */

#include <iostream>
#include "cservoarm.h"
#include "uservice.h"

// Global instance visible to the rest of the software
CServoArm servoArm;

void CServoArm::setup()
{
    // If no config exists yet, write defaults to robot.ini
    if (not ini.has("servoarm"))
    {
        ini["servoarm"]["up_position"]   = "-900";
        ini["servoarm"]["mid_position"]  = "-250";
        ini["servoarm"]["down_position"] = "120";
        ini["servoarm"]["velocity"]      = "200";
        ini["servoarm"]["servo_idx"]     = "1";
    }

    // Read values from robot.ini
    arm_up_pos   = strtol(ini["servoarm"]["up_position"].c_str(),   nullptr, 10);
    arm_mid_pos  = strtol(ini["servoarm"]["mid_position"].c_str(),  nullptr, 10);
    arm_down_pos = strtol(ini["servoarm"]["down_position"].c_str(), nullptr, 10);
    arm_velocity = strtol(ini["servoarm"]["velocity"].c_str(),      nullptr, 10);
    servo_idx    = strtol(ini["servoarm"]["servo_idx"].c_str(),     nullptr, 10);

    std::cout << "[ServoArm] Setup complete\n";
    std::cout << "[ServoArm] up_pos="   << arm_up_pos
              << " mid_pos="            << arm_mid_pos
              << " down_pos="           << arm_down_pos
              << " velocity="           << arm_velocity
              << " servo_idx="          << servo_idx << "\n";
}

void CServoArm::moveUp()
{
    std::cout << "[ServoArm] Moving UP\n";
    servo[TEENSY_NUM].setServo(servo_idx, true, arm_up_pos, arm_velocity);
}

void CServoArm::moveMid()
{
    std::cout << "[ServoArm] Moving MID\n";
    servo[TEENSY_NUM].setServo(servo_idx, true, arm_mid_pos, arm_velocity);
}

void CServoArm::moveDown()
{
    std::cout << "[ServoArm] Moving DOWN\n";
    servo[TEENSY_NUM].setServo(servo_idx, true, arm_down_pos, arm_velocity);
}

void CServoArm::tick()
{
    servo[TEENSY_NUM].tick();
}