/**
 * @file LQR.cpp
 * @author lsvng
 * @brief Infinite-horizon Linear Quadratic Regulator
 * 
 */

#include <LQR.h>
#include <Eigen/Dense>

using namespace Optimal_Controller;

LQR::LQR(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const int n)
  : Q(Q)
  , R(R)
  , N(n)
{
  statespace = new StateSpaceModel(N);
  P.setZero(Q.rows(), Q.cols());
}

LQR::~LQR()
{
  delete statespace;
}

Eigen::MatrixXd LQR::getControlInput(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C, const Eigen::MatrixXd& E)
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

  this->computeHamiltonian(A, B);
  this->computeRiccati(A, B, E);

  return controlInput;
}

void LQR::computeHamiltonian(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
  // Set Hamilton matrix.
  Eigen::MatrixXd Ham;
  Ham = Eigen::MatrixXd::Zero(2 * A.rows(), 2 * A.rows());
  Ham << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();

  // Get eigenvalues and eigenvectors from Hamilton matrix.
  Eigen::EigenSolver<Eigen::MatrixXd> eigen(Ham);

  // Form a 2nxn matrix whos columns from a basis of the corresponding subspace.
  eigenVec = Eigen::MatrixXcd::Zero(2 * A.rows(), A.rows());

  // Iterate over the Hamilton matrix and extract a stable Eigenvector.
  int j = 0;
  for(unsigned int i = 0; i < 2 * A.rows(); ++i)
  {
    // Get the negative real part: a stable value in the Laplace domain.
    if(eigen.eigenvalues()[i].real() < 0)
    {
      eigenVec.col(j) = eigen.eigenvectors().block(0, i, 2 * A.rows(), 1);
      ++j;
    }
  }
}

void LQR::computeRiccati(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& E)
{
  Eigen::MatrixXd K;
  Eigen::MatrixXcd U11, U21;

  U11 = eigenVec.block(0, 0, A.rows(), A.rows());
  U21 = eigenVec.block(A.rows(), 0, A.rows(), A.rows());
  P = (U21 * U11.inverse()).real();

  // Update LQR gain.
  K = R.inverse() * B.transpose() * P;
  controlInput = K * E;
}
