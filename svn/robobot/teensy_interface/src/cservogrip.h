/*  
 * cservogrip.h
 * 
 * High-level interface for gripper servo control.
 * Other modules can include this and call openGrip() / closeGrip() directly.
 * 
 * Gripper configuration values are read from robot.ini under [servogrip].
 * To tune the gripper, only change values in robot.ini:
 *     [servogrip]
 *     open_position  = 200
 *     mid_position   = 0
 *     close_position = -200
 *     velocity       = 200
 *     servo_idx      = 2
 *
 * Copyright © 2023 DTU, Christian Andersen jcan@dtu.dk
 * The MIT License (MIT)  https://mit-license.org/
 */

#ifndef CSERVOGRIP_H
#define CSERVOGRIP_H

#include "cservo.h"

class CServoGrip
{
public:
    /**
     * Setup the servo gripper - call once at startup.
     * Reads configuration from robot.ini under [servogrip].
     * If no config exists, default values are written to robot.ini.
     */
    void setup();

    /**
     * Move gripper to full open position.
     */
    void openGrip();

    /**
     * Move gripper to the middle position.
     */
    void midGrip();

    /**
     * Move gripper to full closed position.
     */
    void closeGrip();

    /**
     * Keep data flow alive - called in main loop.
     */
    void tick();

private:
    int grip_open_pos  = 200;
    int grip_mid_pos   = 0;
    int grip_close_pos = -200;
    int grip_velocity  = 200;
    int servo_idx      = 2;

    static const int TEENSY_NUM = 0;
};

/**
 * Make this visible to the rest of the software
 */
extern CServoGrip servoGrip;

#endif