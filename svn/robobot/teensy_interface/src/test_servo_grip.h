#ifndef TEST_SERVO_GRIP_H
#define TEST_SERVO_GRIP_H

/**
 * Runs the manual servo gripper test loop.
 * Called from main() when a gripper test flag is passed.
 */
void testGripLoop();

/**
 * Command the gripper to open, close, or move to the middle position.
 *
 * \param action 'o' = open, 'c' = close, 'm' = middle
 */
void commandGrip(char action);

#endif