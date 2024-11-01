/**
 * @file LQG.h
 * @author lsvng
 * @brief Linear–quadratic–Gaussian control
 * 
 */

#ifndef LQG_H
#define LQG_H

#include <Eigen/Dense>
#include <StateSpaceModel.h>

namespace Optimal_Controller
{
  class LQG
  {
    public:
      /**
       * @brief Construct a new LQG object
       * 
       * @param Q Weight on the systems state
       * @param R Weight on control input
       * @param n number of states
       * 
       */
      LQG(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const int n);

      /**
       * @brief Destroy the LQG object
       * 
       */
      ~LQG();

      /**
       * @brief Disable copy constructor
       * 
       * @param iLQG 
       */
      LQG(LQG& iLQG) = delete;

      /**
       * @brief Disable assignment constructor
       * 
       * @param iLQG 
       */
      void operator=(LQG& iLQG) = delete;

      /**
       * @brief Get the Control Input
       * 
       * @param iA System dynamics matrix
       * @param iB Input matrix
       * @param iC Output matrix
       * @param setpoint Setpoint
       * @param measurement Current measurement
       * @return Eigen::MatrixXd 
       */
      Eigen::MatrixXd getControlInput(const Eigen::MatrixXd& iA, const Eigen::MatrixXd& iB, const Eigen::MatrixXd& iC, const Eigen::VectorXd& setpoint, const Eigen::VectorXd& measurement);

    private:
    /**
     * @brief To get positive-definite matrix that represents the steady-state solution, P
     * 
     */
      void schurDecomposition();

      /**
       * @brief 
       * 
       * @param C Output matrix
       * @param setpoint Setpoint
       * @param measurement Current measurement
       */
      void computeRiccati(const Eigen::MatrixXd& C, const Eigen::VectorXd& setpoint, const Eigen::VectorXd& measurement);

    private:
      Eigen::MatrixXd A;            // System dynamics matrix
      Eigen::MatrixXd B;            // Input matrix
      Eigen::MatrixXd Q;            // Weight on the systems state
      Eigen::MatrixXd R;            // Weight on control input
      Eigen::VectorXd controlInput; // Control input
      Eigen::VectorXd X_hat;        // State estimate
      Eigen::MatrixXd P_new;        // Covariance matrix
      
      int N;                        // Number of states
      StateSpaceModel* statespace;  // State Space Model object
  };
}

#endif /* LQG_H */
