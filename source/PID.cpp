/**
 * @file pid_controller.cpp
 * @author lsvng
 * @brief PID Controller Lib.
 * 
 */

#include <PID.h>
#include <chrono>

using namespace Optimal_Controller;

PID::PID(const float iKp, const float iKi, const float iKd, const float iMin, const float iMax) 
  : kp(iKp)
  , ki(iKi)
  , kd(iKd)
  , min(iMin)
  , max(iMax)
{
}

PID::~PID()
{
}

float PID::compute(const float setpoint, const float current_state, const float dt)
{
  static float previous_error;
  static float integral;
  float proportional_error;
  float derivative_error;
  float proportional;
  float derivative;
  float output;

  // Compute state error
  proportional_error = setpoint - current_state;
  integral           = (integral + proportional_error) * dt;
  derivative_error   = (proportional_error - previous_error) / dt;

  // Compute PID
  proportional = kp * proportional_error;
  integral     = ki * integral;
  derivative   = kd * derivative_error;

  // Compute desired state
  output = proportional + integral + derivative;

  // Hard constraints
  if(output > max) 
  {
    output = max;
  }
  else if (output < min) 
  {
    output = min;
  }

  // Update previous error
  previous_error = proportional_error;

  return output;
}
