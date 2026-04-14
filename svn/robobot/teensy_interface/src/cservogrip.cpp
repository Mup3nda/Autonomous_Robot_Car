/*  
 * cservogrip.cpp
 * 
 * High-level interface for gripper servo control.
 */

#include <iostream>
#include "cservogrip.h"
#include "uservice.h"

// Global instance visible to the rest of the software
CServoGrip servoGrip;

void CServoGrip::setup()
{
    if (not ini.has("servogrip"))
    {
        ini["servogrip"]["open_position"]  = "-900";
        ini["servogrip"]["mid_position"]   = "-500";
        ini["servogrip"]["close_position"] = "-400";
        ini["servogrip"]["velocity"]       = "200";
        ini["servogrip"]["servo_idx"]      = "2";
    }

    grip_open_pos  = strtol(ini["servogrip"]["open_position"].c_str(),  nullptr, 10);
    grip_mid_pos   = strtol(ini["servogrip"]["mid_position"].c_str(),   nullptr, 10);
    grip_close_pos = strtol(ini["servogrip"]["close_position"].c_str(), nullptr, 10);
    grip_velocity  = strtol(ini["servogrip"]["velocity"].c_str(),       nullptr, 10);
    servo_idx      = strtol(ini["servogrip"]["servo_idx"].c_str(),      nullptr, 10);

    printf("# ServoGrip setup complete\n");
    printf("# ServoGrip open_pos=%d mid_pos=%d close_pos=%d velocity=%d servo_idx=%d\n",
           grip_open_pos, grip_mid_pos, grip_close_pos, grip_velocity, servo_idx);
}

void CServoGrip::openGrip()
{
    printf("# ServoGrip opening\n");
    servo[TEENSY_NUM].setServo(servo_idx, true, grip_open_pos, grip_velocity);
}

void CServoGrip::midGrip()
{
    printf("# ServoGrip moving MID\n");
    servo[TEENSY_NUM].setServo(servo_idx, true, grip_mid_pos, grip_velocity);
}

void CServoGrip::closeGrip()
{
    printf("# ServoGrip closing\n");
    servo[TEENSY_NUM].setServo(servo_idx, true, grip_close_pos, grip_velocity);
}

void CServoGrip::tick()
{
    servo[TEENSY_NUM].tick();
}