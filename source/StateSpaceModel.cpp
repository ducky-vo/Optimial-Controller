/**
 * @file StateSpaceModel.cpp
 * @author duckstarr
 * @brief State Space Model
 * 
 */

#include <StateSpaceModel.h>
#include <unsupported/Eigen/MatrixFunctions>

namespace Optimal_Controller
{
  StateSpaceModel::StateSpaceModel(uint8_t n)
    : n(n)
  {
    mStateSpace.setZero(n, n); // [B AB A^2B ... A^n-1B] || [C^T A^TC^T ... (A^T)^n-1C^T]
  }

  StateSpaceModel::~StateSpaceModel()
  {
  }

  bool StateSpaceModel::isControllable(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
  {
    return this->computeDeterminant(A, B) != 0 ? true : false;
  }

  bool StateSpaceModel::isObservable(const Eigen::MatrixXd& A, const Eigen::MatrixXd& C)
  {
    Eigen::MatrixXd wA;
    Eigen::MatrixXd wC;

    wA = A.transpose();
    wC = C.transpose();

    return this->computeDeterminant(wA, wC) != 0 ? true : false;
  }

  double StateSpaceModel::computeDeterminant(const Eigen::MatrixXd& A, const Eigen::MatrixXd& MAT)
  {
    if (A.cols() == MAT.rows())
    {
      Eigen::MatrixPower<Eigen::MatrixXd> Apow(A);
      
      for (int i = 0; i < n; i++)
      {
        mStateSpace.col(i) = Apow(i) * MAT;
      }
    }

    return mStateSpace.determinant();
  }
}
