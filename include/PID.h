/**
 * @file PID.h
 * @author lsvng
 * @brief PID Controller.
 */

#ifndef PID_H
#define PID_H

#include <vector>

namespace Optimal_Controller
{

class PID
{
  public:
    /**
     * @brief A PID controller that continuously calculates an error value and adjust them accordingly.
     * 
     * @param kp proportional gain.
     * @param ki integral gain.
     * @param kd derivative gain.
     * @param iMin minimum value of manipulated variable.
     * @param iMax maximum value of manipulated variable.
     */
    PID(const float iKp, const float iKi, const float iKd, const float iMin, const float iMax);

    /**
     * @brief Destroy the PID object
     * 
     */
    ~PID();

    /**
     * @brief Disable copy constructor
     * 
     * @param iPID 
     */
    PID(PID & iPID) = delete;

    /**
     * @brief Disable assignment constructor
     * 
     * @param iPID 
     */
    void operator=(PID& iPID) = delete;

    /**
     * @brief Compute the desired state of a system.
     * 
     * @param setpoint the goal of the system. Example: to go from point A to B, the setpoint is B.
     * @param current_state the current state of a system. Example: to go from A to B, the current_state is A.
     * @param dt the time difference between two data sets. Example: the time it takes to go from A to A + 1 is dt.
     * @return float 
     */
    float compute(const float setpoint, const float current_state, const float dt);

  private:
    float min; // Min value of the output state   
    float max; // Max value of the output state
    float kp;  // Proportional gain
    float ki;  // Integral gain
    float kd;  // Derivative gain
};

} // namespace controller

#endif /* PID_H */
