/*
 #***************************************************************************
 #*   Copyright (C) 2024 by DTU
 #*   jcan@dtu.dk
 #*
 #* The MIT License (MIT)  https://mit-license.org/
 */

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <ostream>
#include <chrono>
#include <thread>
#include "uservice.h"
#include "cmixer.h"
#include "sgpiod.h"
#include "utime.h"
#include "cservo.h"
#include "cservoarm.h"
#include "scurrent.h"
#include "sdistforce.h"
#include "sedge.h"
#include "sencoder.h"
#include "simu.h"
#include "umqttin.h"
#include "test_servo_arm.h"

void loop()
{
    teensy[0].send("leds 14 0 45 0\n", true);
    int g = 5;
    int dg = 25;
    const int MSL = 50;
    char s[MSL];
    UTime t("now");
    while (not service.stopNowRequest)
    {
        usleep(50000);
        if (t.getTimePassed() > 0.2)
        {
            t.now();
            g += dg;
            if (g > 100 or g <= abs(dg))
                dg = -dg;
            snprintf(s, MSL, "leds 14 0 %d 0\n", g);
            teensy[0].send(s, true);
            for (int i = 0; i < NUM_TEENSY_MAX; i++)
            {
                servo[i].tick();
                current[i].tick();
                distforce[i].tick();
                edge[i].tick();
                encoder[i].tick();
                imu[i].tick();
                robot[i].tick();
                teensy[i].tick();
                motor[i].tick();
                mqttin.tick();
            }
        }
    }
    teensy[0].send("leds 14 0 0 0\n", true);
}

int main (int argc, char **argv)
{
    // Check for test mode flag before anything else
    bool testServoMode = false;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--test-servo") == 0)
        {
            testServoMode = true;
            for (int j = i; j < argc - 1; j++)
                argv[j] = argv[j + 1];
            argc--;
            break;
        }
    }

    if (testServoMode)
    {
        std::cout << "[MAIN] Running in servo test mode\n";
        service.setup(argc, argv);
        if (not service.theEnd)
        {
            servoArm.setup();
            testServoLoop();
        }
        service.terminate();
        exit(0);
    }

    // Normal startup
    int a = service.isThisProcessRunning("teensy_interfac");
    if (a == 1)
    {
        service.setup(argc, argv);
        if (not service.theEnd)
        {
            servoArm.setup();
            loop();
        }
        service.terminate();
        printf("# ---- Teensy_interface has ended (nicely) ----\r\n");
    }
    else
        printf("# ---- Teensy_interface is running already (stop with 'pkill teensy_interfac' ----\r\n");

    exit(0);
}