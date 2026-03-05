/*
 * test_servo_arm.cpp
 *
 * Temporary manual test for the 1DOF arm servo.
 * Stands in for the ball detection trigger while that module is being developed.
 *
 * When ball detection is ready, replace the manual input loop with:
 *     bool ballDetected = (ballDistance < THRESHOLD);
 *     commandArm(ballDetected);
 */

#include <iostream>
#include "cservo.h"
#include "utime.h"
#include "test_servo_arm.h"

// -------------------------------------------------------
// Arm configuration - tune these to your physical setup
// -------------------------------------------------------
#define ARM_UP_POSITION      0      // Upright / resting position
#define ARM_DOWN_POSITION    400    // Down / deployed position
#define ARM_90_UP_POSITION   250    // 90 degrees upward  - tune this
#define ARM_90_DOWN_POSITION -250   // 90 degrees downward - tune this
#define ARM_SERVO_NUM        1      // Servo number the arm is attached to
#define ARM_VELOCITY         200    // Movement speed in servo units/sec
#define TEENSY_NUM           0      // Teensy 0 confirmed

// -------------------------------------------------------
// State tracking
// -------------------------------------------------------
static bool armIsDown = false;

// -------------------------------------------------------
// commandArm()
// This is the function the upper layer will eventually call.
// For now triggered manually via keyboard input.
// -------------------------------------------------------
void commandArm(bool goDown)
{
    if (goDown && !armIsDown)
    {
        std::cout << "[ARM] Moving DOWN\n";
        servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_DOWN_POSITION, ARM_VELOCITY);
        armIsDown = true;
    }
    else if (!goDown && armIsDown)
    {
        std::cout << "[ARM] Moving UP\n";
        servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
        armIsDown = false;
    }
    else
    {
        std::cout << "[ARM] Already in requested position, no movement needed\n";
    }
}

// -------------------------------------------------------
// printServoStatus()
// -------------------------------------------------------
void printServoStatus()
{
    std::cout << "\n--- Servo Status ---\n";
    std::cout << "Arm state     : " << (armIsDown ? "DOWN" : "UP")                         << "\n";
    std::cout << "Servo ref pos : " << servo[TEENSY_NUM].servo_ref[ARM_SERVO_NUM - 1]      << "\n";
    std::cout << "Servo act pos : " << servo[TEENSY_NUM].servo_position[ARM_SERVO_NUM - 1] << "\n";
    std::cout << "Servo enabled : " << servo[TEENSY_NUM].servo_enabled[ARM_SERVO_NUM - 1]  << "\n";
    std::cout << "Update count  : " << servo[TEENSY_NUM].updateCnt                         << "\n";
    std::cout << "--------------------\n\n";
}

// -------------------------------------------------------
// printServoHelp()
// -------------------------------------------------------
void printServoHelp()
{
    std::cout << "\n=============================\n";
    std::cout << " Servo Arm Manual Test\n";
    std::cout << "=============================\n";
    std::cout << " [d] - Move arm DOWN (full)\n";
    std::cout << " [u] - Move arm UP (full)\n";
    std::cout << " [1] - Move arm 90 degrees UP\n";
    std::cout << " [2] - Move arm 90 degrees DOWN\n";
    std::cout << " [t] - Toggle arm position\n";
    std::cout << " [s] - Print servo status\n";
    std::cout << " [e] - Disable servo\n";
    std::cout << " [h] - Print this help\n";
    std::cout << " [q] - Quit\n";
    std::cout << "=============================\n\n";
}

// -------------------------------------------------------
// testServoLoop()
// -------------------------------------------------------
void testServoLoop()
{
    servo[TEENSY_NUM].setup(TEENSY_NUM);

    printServoHelp();

    std::cout << "[INIT] Setting arm to UP position\n";
    servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
    armIsDown = false;

    char input;
    bool running = true;

    while (running)
    {
        servo[TEENSY_NUM].tick();

        std::cout << "Enter command: ";

        if (!(std::cin >> input))
            break;

        switch (input)
        {
            case 'd':
                commandArm(true);
                break;

            case 'u':
                commandArm(false);
                break;

            case '1':
                std::cout << "[ARM] Moving to 90 degrees UP\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_90_UP_POSITION, ARM_VELOCITY);
                break;

            case '2':
                std::cout << "[ARM] Moving to 90 degrees DOWN\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_90_DOWN_POSITION, ARM_VELOCITY);
                break;

            case 't':
                std::cout << "[TEST] Toggling arm\n";
                commandArm(!armIsDown);
                break;

            case 's':
                printServoStatus();
                break;

            case 'e':
                std::cout << "[TEST] Disabling servo\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, false, ARM_UP_POSITION, 0);
                armIsDown = false;
                break;

            case 'h':
                printServoHelp();
                break;

            case 'q':
                std::cout << "[TEST] Quitting - returning arm to UP position\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
                running = false;
                break;

            default:
                std::cout << "[TEST] Unknown command. Press [h] for help.\n";
                break;
        }
    }

    servo[TEENSY_NUM].terminate();
    std::cout << "[TEST] Terminated cleanly\n";
}