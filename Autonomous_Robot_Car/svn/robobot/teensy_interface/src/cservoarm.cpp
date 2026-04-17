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
    if (not ini.has("servoarm"))
    {
        ini["servoarm"]["up_position"]   = "-900";
        ini["servoarm"]["mid_position"]  = "-250";
        ini["servoarm"]["down_position"] = "120";
        ini["servoarm"]["velocity"]      = "200";
        ini["servoarm"]["servo_idx"]     = "1";
    }

    arm_up_pos   = strtol(ini["servoarm"]["up_position"].c_str(),   nullptr, 10);
    arm_mid_pos  = strtol(ini["servoarm"]["mid_position"].c_str(),  nullptr, 10);
    arm_down_pos = strtol(ini["servoarm"]["down_position"].c_str(), nullptr, 10);
    arm_velocity = strtol(ini["servoarm"]["velocity"].c_str(),      nullptr, 10);
    servo_idx    = strtol(ini["servoarm"]["servo_idx"].c_str(),     nullptr, 10);

    printf("# ServoArm setup complete\n");
    printf("# ServoArm up_pos=%d mid_pos=%d down_pos=%d velocity=%d servo_idx=%d\n",
           arm_up_pos, arm_mid_pos, arm_down_pos, arm_velocity, servo_idx);
}

void CServoArm::moveUp()
{
    printf("# ServoArm moving UP\n");
    servo[TEENSY_NUM].setServo(servo_idx, true, arm_up_pos, arm_velocity);
}

void CServoArm::moveDown()
{
    printf("# ServoArm moving DOWN\n");
    servo[TEENSY_NUM].setServo(servo_idx, true, arm_down_pos, arm_velocity);
}

void CServoArm::moveMid()
{
    printf("# ServoArm moving MID\n");
    servo[TEENSY_NUM].setServo(servo_idx, true, arm_mid_pos, arm_velocity);
}

void CServoArm::tick()
{
    servo[TEENSY_NUM].tick();
}