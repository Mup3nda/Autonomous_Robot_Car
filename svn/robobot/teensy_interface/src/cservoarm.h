/*  
 * cservoarm.h
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

#ifndef CSERVOARM_H
#define CSERVOARM_H

#include "cservo.h"

class CServoArm
{
public:
    /**
     * Setup the servo arm - call once at startup.
     * Reads configuration from robot.ini under [servoarm].
     * If no config exists, default values are written to robot.ini.
     */
    void setup();

    /**
     * Move arm to full upright/resting position.
     * Call this when arm is not needed or no ball is detected.
     */
    void moveUp();

    /**
     * Move arm to mid position.
     * Position configured in robot.ini (mid_position).
     */
    void moveMid();

    /**
     * Move arm to full down/deployed position.
     * Call this when ball is detected.
     */
    void moveDown();

    /**
     * Keep data flow alive - called in main loop.
     */
    void tick();

private:
    // Values loaded from robot.ini at setup()
    int arm_up_pos   = -900;
    int arm_mid_pos  = -250;
    int arm_down_pos = 120;
    int arm_velocity = 200;
    int servo_idx    = 1;

    // Hardware constant - not configurable
    static const int TEENSY_NUM = 0;
};

/**
 * Make this visible to the rest of the software
 */
extern CServoArm servoArm;

#endif