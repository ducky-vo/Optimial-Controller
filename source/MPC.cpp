/**
 * @file MPC.cpp
 * @author lsvng
 * @brief Finite-horizon Model Predictive Controller
 * 
 */

#include <MPC.h>
#include <Eigen/Dense>

using namespace Optimal_Controller;

MPC::MPC(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const double S, const double tolarance, const int n) 
  : Q(Q)
  , R(R)
  , saturation(S)
  , tolarance(tolarance)
  , N(n)
  , converged(false)
{
  statespace = new StateSpaceModel(N);

  Ad.setZero(Q.rows(), Q.cols());
  I.setIdentity(Q.rows(), Q.cols());
}

MPC::~MPC()
{
  delete statespace;
}

Eigen::MatrixXd MPC::getControlInput(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C, const Eigen::MatrixXd& E, double dt)
{
  if (!statespace->isControllable(A, B) || !statespace->isObservable(A, C))
  {
    printf("State-Space model is NOT reliable.\n");
    return controlInput.setZero(1, A.cols());
  }

  if ((C.transpose() * C).rows() != Q.rows() || R.cols() != B.cols())
  {
    printf("Weight matricies is NOT compatible with state-space model.\n");
    return controlInput.setZero(1, A.cols());
  }

  this->discretization(A, B, dt);
  this->computeRiccati(E, dt);

  return controlInput;
}

void MPC::discretization(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double dt)
{
  Ad = (I + A * dt * 0.5) * (I - A * dt * 0.5).inverse();
  Bd = B * dt;
}

void MPC::computeRiccati(const Eigen::MatrixXd& E, double dt)
{
  Eigen::MatrixXd K;
  Eigen::MatrixXd P_new;

  // A Cost-to-go matrix P evolving backwards in time from Q is defined as,
  P = Q;

  // Iterating over a finite-horizon Algebraic Riccati Equation to a steady-state solution; that is, P_new - P ~= 0 + tolarance
  for(unsigned int i = 0; i < numIterations; i++)
  {
    P_new = Ad.transpose() * P * Ad - (Ad.transpose() * P * Bd) * (R + Bd.transpose() * P * Bd).inverse() * (Bd.transpose() * P * Ad) + Q;

    if(fabs((P_new - P).maxCoeff()) < tolarance)
    {
      converged = true;
      break;
    }

    // Update Riccati solution
    P = P_new;
  }

    K = (R + Bd.transpose() * P * Bd).inverse() * Bd.transpose() * P * Ad;
    controlInput = K * E;
}
