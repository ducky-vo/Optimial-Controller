/**
 * @file PID.cpp
 * @brief !Valgrind output
 * Memcheck, a memory error detector
 * Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
 * Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
 * Command: ./OPTIMAL_CONTROLLER_PID
 * 
 * 
 * HEAP SUMMARY:
 *     in use at exit: 0 bytes in 0 blocks
 *   total heap usage: 3 allocs, 3 frees, 73,748 bytes allocated
 * 
 * All heap blocks were freed -- no leaks are possible
 * 
 * For lists of detected and suppressed errors, rerun with: -s
 * ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
 */

#include <PID.h>
#include <memory>
#include <iostream>

using namespace Optimal_Controller;

int main()
{
  double kp, ki, kd;
  double min, max;
  float x, y;
  float dt;

  kp = 1.0; // Proportional gain
  ki = 0.2; // Integral gain
  kd = 0.0; // Derivative gain

  min = -5.0; // Min velocity
  max = 5.0;  // Max velocity

  x = 4.0; // X Position
  y = 0.0; // Y Position

  dt = 0.1; // Delta time

  auto ptr = std::make_unique<PID>(kp, ki, kd, min, max);

  auto velocity = ptr->compute(x, y, dt);

  /**
   * @brief velocity: 4.08
   */
  std::cout << "velocity: " << velocity << std::endl;

  return EXIT_SUCCESS;
}
