/**
 * @file LQG.cpp
 * @author lsvng
 * @brief Linear–quadratic–Gaussian control
 * 
 */

#include <LQG.h>
#include <Eigen/Dense>

using namespace Optimal_Controller;

LQG::LQG(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const int n)
  : Q(Q)
  , R(R)
  , N(n)
{
  statespace = new StateSpaceModel(N);

  controlInput.setZero(N);
  X_hat.setZero(N);
}

LQG::~LQG()
{
  delete statespace;
}

Eigen::MatrixXd LQG::getControlInput(const Eigen::MatrixXd& iA, const Eigen::MatrixXd& iB, const Eigen::MatrixXd& iC, const Eigen::VectorXd& setpoint, const Eigen::VectorXd& measurement)
{
  if (!statespace->isControllable(iA, iB) || !statespace->isObservable(iA, iC))
  {
    printf("State-Space model is NOT reliable.\n");
    return controlInput.setZero(iA.cols());
  }

  if ((iC.transpose() * iC).rows() != Q.rows() || R.cols() != iB.cols())
  {
    printf("Weight matricies is NOT compatible with state-space model.\n");
    return controlInput.setZero(iA.cols());
  }

  A = iA;
  B = iB;

  this->schurDecomposition();
  this->computeRiccati(iC, setpoint, measurement);

  return controlInput;
}

void LQG::schurDecomposition()
{
  Eigen::MatrixXd P;
  Eigen::MatrixXd U;
  Eigen::MatrixXd T;
  Eigen::RealSchur<Eigen::MatrixXd> schur(A); // schur decomposition

  if (schur.info() != Eigen::Success) 
  {
    printf("Schur decomposition failed to converge!\n");
    return;
  }

  U = schur.matrixU(); // Get the unitary matrix U
  T = schur.matrixT(); // Get the upper triangular matrix T
  
  if (U.size() != T.size())
  {
    printf("The unitary matrix is not the same size as the triangular matrix\n");
    return;
  }

  P = U * T.diagonal().array().exp().matrix().asDiagonal() * U.adjoint();
  P_new = A.transpose() * P * A - A.transpose() * P * B * (R + B.transpose() * P * B).inverse() * B.transpose() * P * A + Q;
}

void LQG::computeRiccati(const Eigen::MatrixXd& C, const Eigen::VectorXd& setpoint, const Eigen::VectorXd& measurement)
{
  Eigen::MatrixXd L;
  Eigen::MatrixXd K;

  if (C.rows() != measurement.rows() || B.cols() != setpoint.rows())
  {
    printf("System dimension is not set properly\n");
    return;
  }

  L = P_new * C.transpose() * (C * P_new * C.transpose() + R).inverse();
  K = (R + B.transpose() * P_new * B).inverse() * B.transpose() * P_new * A;
  
  X_hat = A * X_hat + B * setpoint - L * (C * X_hat - measurement);
  controlInput = -K * X_hat;
}
