/**
 * @file MPC.h
 * @author lsvng
 * @brief Finite-horizon Model Predictive Controller
 * 
 */

#ifndef MPC_H
#define MPC_H

#include <Eigen/Dense>
#include <StateSpaceModel.h>

namespace Optimal_Controller
{
  namespace
  {
    const unsigned int numIterations = 100000;
  }
  
  class MPC
  {
    public:
      /**
       * @brief An MPC based on the following problem statement:
       * 
       *        min(U) with penalties on the current state, final state, and input signal (Q, P, and R, respectively).
       * 
       *        s.t. x0 = x(t)
       *             x(k + 1) = Ax(k) + Bu(k), k = 0, ... , N-1
       *             u_min <= u <= u_max
       *             y˙ cos θ − x˙ sin θ = 0 non-holonomic constraint
       *             u(k) ∈ U if sensor is in range.
       *             x(k) ∈ X if sensor is in range.
       * 
       * @param Q Weight on the systems state
       * @param R Weight on control input
       * @param S Saturation
       * @param tolarance Tolarance
       * @param n number of states
       * 
       */
      MPC(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const double S, const double tolarance, const int n);
      
      /**
       * @brief Destroy the MPC object
       * 
       */
      ~MPC();
      
      /**
       * @brief Disable copy constructor
       * 
       * @param iMPC 
       */
      MPC(MPC& iMPC) = delete;

      /**
       * @brief Disable assignment constructor
       * 
       * @param iMPC 
       */
      void operator=(MPC& iMPC) = delete;

      /**
       * @brief Get the Control Input
       * 
       * @param A System dynamics matrix
       * @param B Input matrix
       * @param B Output matrix
       * @param E State error
       * @param dt Delta time
       * @return Eigen::MatrixXd 
       */
      Eigen::MatrixXd getControlInput(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C, const Eigen::MatrixXd& E, double dt);

    private:
      /**
       * @brief Solve Riccati Equation
       * 
       * @param E State error
       * @param dt Delta time
       * 
       */
      void computeRiccati(const Eigen::MatrixXd& E, double dt);

      /**
       * @brief Compute velocity
       * 
       */
      void computeVelocity();

      /**
       * @brief 
       * 
       * @param A System dynamics matrix
       * @param B Input matrix
       * @param dt Delta time
       */
      void discretization(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double dt);

    private:
      Eigen::MatrixXd Q;            // Weight on the systems state
      Eigen::MatrixXd R;            // Weight on control input
      Eigen::MatrixXd I;            // Identity matrix
      Eigen::MatrixXd P;            // Riccati matrix
      Eigen::MatrixXd Ad;           // Discretized system dynamics matrix
      Eigen::MatrixXd Bd;           // Discretized input matrix
      Eigen::MatrixXd controlInput; // Control input
      
      double saturation;            // System saturation
      double tolarance;             // System tolarance
      int N;                        // Number of states
      bool converged;               // Check system convergence in the S-plane

      StateSpaceModel* statespace;  // State Space Model object
  };

} // namespace controller

#endif /* MPC_H */
