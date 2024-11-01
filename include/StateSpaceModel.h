/**
 * @file StateSpaceModel.h
 * @author lsvng
 * @brief State Space Model
 * 
 */

#ifndef STATESPACEMODEL_H
#define STATESPACEMODEL_H

#include <Eigen/Dense>
#include <Matrix.h>

namespace Optimal_Controller
{
  class StateSpaceModel
  {
    public:
      /**
       * @brief Construct a new StateSpaceModel object
       * 
       * @param n number of states
       */
      StateSpaceModel(uint8_t n);

      /**
       * @brief Destroy the StateSpaceModel object
       * 
       */
      ~StateSpaceModel();
      
      /**
       * @brief Disable copy operator
       * 
       * @param iStateSpaceModel
       */
      StateSpaceModel(StateSpaceModel& iStateSpaceModel) = delete;

      /**
       * @brief Disable assignment constructor
       * 
       * @param iStateSpaceModel 
       */
      void operator=(StateSpaceModel& iStateSpaceModel) = delete;

      /**
       * @brief Check of state-space model is controllable
       * 
       * @param A System dynamics matrix
       * @param B Input matrix
       * @return true 
       * @return false 
       */
      bool isControllable(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

      /**
       * @brief Check of state-space model is observable
       * 
       * @param A System dynamics matrix
       * @param C Output matrix
       * @return true 
       * @return false 
       */
      bool isObservable(const Eigen::MatrixXd& A, const Eigen::MatrixXd& C);

    private:
      /**
       * @brief Compute the number of states from state-space model
       * 
       * @param A 
       * @return uint8_t 
       */
      uint8_t computeNumberOfStates(const Eigen::MatrixXd& A);

      /**
       * @brief Compute the determinant of StateSpaceModel matrix
       * 
       * @param A System dynamics matrix
       * @param MAT Input matrix / Output matrix
       * @return double 
       */
      double computeDeterminant(const Eigen::MatrixXd& A, const Eigen::MatrixXd& MAT);

    private:
      Eigen::MatrixXd mStateSpace; // StateSpaceModel matrix: [B AB A^2B ... A^n-1B]
      uint8_t n;                   // number of state (i.e., x, x_dot, theta, theta_dot)
  };
}

#endif // STATESPACEMODEL_H
