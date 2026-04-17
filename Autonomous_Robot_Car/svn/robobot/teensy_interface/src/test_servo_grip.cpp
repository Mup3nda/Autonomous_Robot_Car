/*
 * test_servo_grip.cpp
 *
 * Temporary manual test for the 1DOF gripper servo.
 */

#include <iostream>
#include "cservogrip.h"
#include "uservice.h"
#include "test_servo_grip.h"

void commandGrip(char action)
{
    if (action == 'o')
    {
        std::cout << "[GRIP] Moving OPEN\n";
        servoGrip.openGrip();
    }
    else if (action == 'c')
    {
        std::cout << "[GRIP] Moving CLOSED\n";
        servoGrip.closeGrip();
    }
    else if (action == 'm')
    {
        std::cout << "[GRIP] Moving to MID position\n";
        servoGrip.midGrip();
    }
    else
    {
        std::cout << "[GRIP] Unknown command\n";
    }
}

void testGripLoop()
{
    servoGrip.setup();

    std::cout << "\n=============================\n";
    std::cout << " Servo Gripper Manual Test\n";
    std::cout << "=============================\n";
    std::cout << " [o] - Open gripper\n";
    std::cout << " [c] - Close gripper\n";
    std::cout << " [m] - Move gripper to MID\n";
    std::cout << " [s] - Show servo status\n";
    std::cout << " [q] - Quit\n";
    std::cout << "=============================\n\n";

    std::cout << "[INIT] Setting gripper to OPEN position\n";
    servoGrip.openGrip();

    FILE* tty = fopen("/dev/tty", "r");
    if (not tty)
    {
        std::cout << "[ERROR] Could not open /dev/tty\n";
        return;
    }

    bool running = true;

    while (running and not service.stop)
    {
        servoGrip.tick();

        std::cout << "Enter command: " << std::flush;

        int c = fgetc(tty);
        if (c == EOF)
            break;

        char input = (char)c;

        if (input == '\n' or input == '\r')
            continue;

        switch (input)
        {
            case 'o':
                commandGrip('o');
                break;

            case 'c':
                commandGrip('c');
                break;

            case 'm':
                commandGrip('m');
                break;

            case 's':
                std::cout << "[STATUS] Gripper uses servo_idx configured in [servogrip]\n";
                break;

            case 'q':
                std::cout << "[TEST] Quitting - returning gripper to OPEN position\n";
                servoGrip.openGrip();
                running = false;
                break;

            default:
                std::cout << "[TEST] Unknown command. Use o, c, m, s, or q.\n";
                break;
        }
    }

    fclose(tty);
    std::cout << "[TEST] Terminated cleanly\n";
}