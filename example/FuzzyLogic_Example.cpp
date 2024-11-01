/**
 * @file mpc.cpp
 * @brief !Valgrind output
 * Memcheck, a memory error detector
 * Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
 * Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
 * Command: ./OPTIMAL_CONTROLLER_FuzzyLogic
 * 
 * 
 * HEAP SUMMARY:
 *     in use at exit: 0 bytes in 0 blocks
 *   total heap usage: 3 allocs, 3 frees, 73,760 bytes allocated
 * 
 * All heap blocks were freed -- no leaks are possible
 * 
 * For lists of detected and suppressed errors, rerun with: -s
 * ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
 */

#include <FuzzyLogic.h>
#include <memory>
#include <iostream>

using namespace Optimal_Controller;

int main()
{
  double min = -5.0;        // Min velocity
  double max = 5.0;         // Max velocity
  double setpoint = 11.1;   // Set point
  double measurement = 3.3; // Current measurement
  double dt = 0.1;          //Delta time

  auto ptr = std::make_unique<FuzzyLogic>(min, max);

  std::cout << "Output: " << ptr->getControlInput(setpoint, measurement, dt) << std::endl;;

  /** !Output
   * @brief Output: 3.9
   * 
   */

  return EXIT_SUCCESS;
}
