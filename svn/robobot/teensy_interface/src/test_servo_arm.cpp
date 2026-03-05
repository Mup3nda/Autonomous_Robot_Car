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
#include "uservice.h"
#include "test_servo_arm.h"

// -------------------------------------------------------
// Arm configuration - tune these to your physical setup
// -------------------------------------------------------
#define ARM_UP_POSITION      400    // Upright / resting position (initial)
#define ARM_DOWN_POSITION    0      // Down / deployed position
#define ARM_90_UP_POSITION   250    // 90 degrees UP - tune this
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
// testServoLoop()
// -------------------------------------------------------
void testServoLoop()
{
    servo[TEENSY_NUM].setup(TEENSY_NUM);

    std::cout << "\n=============================\n";
    std::cout << " Servo Arm Manual Test\n";
    std::cout << "=============================\n";
    std::cout << " [u] - Move arm to initial UP position (" << ARM_UP_POSITION << ")\n";
    std::cout << " [d] - Move arm DOWN\n";
    std::cout << " [2] - Move arm 90 degrees UP\n";
    std::cout << " [s] - Print servo status\n";
    std::cout << " [q] - Quit\n";
    std::cout << "=============================\n\n";

    std::cout << "[INIT] Setting arm to UP position\n";
    servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
    armIsDown = false;

    // Open /dev/tty directly to bypass service keyboard handler
    FILE* tty = fopen("/dev/tty", "r");
    if (!tty)
    {
        std::cout << "[ERROR] Could not open /dev/tty\n";
        return;
    }

    bool running = true;

    while (running and not service.stop)
    {
        servo[TEENSY_NUM].tick();

        std::cout << "Enter command: " << std::flush;

        int c = fgetc(tty);
        if (c == EOF) break;

        char input = (char)c;

        if (input == '\n' or input == '\r')
            continue;

        switch (input)
        {
            case 'u':
                // Always go back to initial position regardless of current state
                std::cout << "[ARM] Returning to initial UP position\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
                armIsDown = false;
                break;

            case 'd':
                std::cout << "[ARM] Moving DOWN\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_DOWN_POSITION, ARM_VELOCITY);
                armIsDown = true;
                break;

            case '2':
                std::cout << "[ARM] Moving to 90 degrees UP\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_90_UP_POSITION, ARM_VELOCITY);
                armIsDown = false; // treat as an intermediate up state
                break;

            case 's':
                printServoStatus();
                break;

            case 'q':
                std::cout << "[TEST] Quitting - returning arm to UP position\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
                running = false;
                break;

            default:
                std::cout << "[TEST] Unknown command. Use u, d, 2, s or q.\n";
                break;
        }
    }

    fclose(tty);
    std::cout << "[TEST] Terminated cleanly\n";
}