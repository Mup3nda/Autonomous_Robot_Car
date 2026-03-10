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
#define ARM_UP_POSITION      0      // Upright / extreme up position
#define ARM_DOWN_POSITION    400    // Down / extreme down position
#define ARM_MID_POSITION     200    // Midpoint / parallel to surface
#define ARM_SERVO_NUM        1      // Servo number the arm is attached to
#define ARM_VELOCITY         200    // Movement speed in servo units/sec
#define TEENSY_NUM           0      // Teensy 0 confirmed

// -------------------------------------------------------
// State tracking
// -------------------------------------------------------
// static bool armIsDown = false; // Note: removed strict state tracking to allow arbitrary transitions between the 3 states

// -------------------------------------------------------
// commandArm()
// This is the function the upper layer will eventually call.
// -------------------------------------------------------
void commandArm(char direction)
{
    if (direction == 'u')
    {
        std::cout << "[ARM] Moving UP (extreme)\n";
        servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
    }
    else if (direction == 'd')
    {
        std::cout << "[ARM] Moving DOWN (extreme)\n";
        servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_DOWN_POSITION, ARM_VELOCITY);
    }
    else if (direction == 'm')
    {
        std::cout << "[ARM] Moving to MID_POINT (parallel)\n";
        servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_MID_POSITION, ARM_VELOCITY);
    }
    else
    {
        std::cout << "[ARM] Unknown command\n";
    }
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
    std::cout << " [u] - Move arm UP (extreme)\n";
    std::cout << " [d] - Move arm DOWN (extreme)\n";
    std::cout << " [m] - Move arm to MID POINT (parallel)\n";
    std::cout << " [s] - Show servo status\n";
    std::cout << " [q] - Quit\n";
    std::cout << "=============================\n\n";

    std::cout << "[INIT] Setting arm to UP position\n";
    servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
    
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
                commandArm('u');
                break;

            case 'd':
                commandArm('d');
                break;

            case 'm':
                commandArm('m');
                break;

            case 's':
                std::cout << "[STATUS] Servo target: " << servo[TEENSY_NUM].servo_ref[ARM_SERVO_NUM-1] 
                          << ", Position: " << servo[TEENSY_NUM].servo_position[ARM_SERVO_NUM-1] << "\n";
                break;

            case 'q':
                std::cout << "[TEST] Quitting - returning arm to UP position\n";
                servo[TEENSY_NUM].setServo(ARM_SERVO_NUM, true, ARM_UP_POSITION, ARM_VELOCITY);
                running = false;
                break;

            default:
                std::cout << "[TEST] Unknown command. Use u, d, m, s, or q.\n";
                break;
        }
    }

    fclose(tty);
    std::cout << "[TEST] Terminated cleanly\n";
}