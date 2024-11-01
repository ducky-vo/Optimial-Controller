/**
 * @file mpc.cpp
 * @brief !Valgrind output
 * Memcheck, a memory error detector
 * Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
 * Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
 * Command: ./OPTIMAL_CONTROLLER_MPC
 * 
 * 
 * HEAP SUMMARY:
 *     in use at exit: 0 bytes in 0 blocks
 *   total heap usage: 42,045 allocs, 42,045 frees, 1,202,720 bytes allocated
 * 
 * All heap blocks were freed -- no leaks are possible
 * 
 * For lists of detected and suppressed errors, rerun with: -s
 * ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
 */

#include <MPC.h>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <math.h>

using namespace Optimal_Controller;

int main()
{
  unsigned int x = 3; // Number of states [x, x_dot, theta, theta_dot]
  unsigned int u = 1; // Input dimension [translation rate]
  unsigned int y = 1; // Output dimension [x, theta]
  double dt = 0.025; // Timestamp
  double saturation = 10;
  double tolarance = 1.E-5;


  // Declare MAT for MPC computation.
  Eigen::MatrixXd A(x, x); // System dynamics matrix.
  Eigen::MatrixXd B(x, u); // Input matrix.
  Eigen::MatrixXd C(y, x); // Output matrix
  Eigen::MatrixXd Q(x, x); // Weight on the systems state.
  Eigen::MatrixXd R(u, u); // Weight on control input.

  A <<
    -0.313, 56.7, 0,
    -0.0139, -0.426, 0,
    0, 56.7, 0;

  B <<
    0.232,
    0.0203,
    0;

  C <<
    0, 0, 1;

  R <<
    1;

  Q <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 2.0;

  auto ptr = std::make_unique<MPC>(Q, R, saturation, tolarance, x);

  // Get state error.
  Eigen::MatrixXd E(x, x);

  E <<
    0.0023, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.001;

  std::cout << "Control Input: " << ptr->getControlInput(A, B, C, E, dt) << std::endl;

  /** !Output.
   * @brief Control Input: -0.00117699           0  0.00139741
   */

  return EXIT_SUCCESS;
}
