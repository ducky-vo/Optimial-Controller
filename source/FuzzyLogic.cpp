/**
 * @file FuzzyLogic.cpp
 * @author lsvng
 * @brief Fuzzy logic controller
 * 
 */

#include <FuzzyLogic.h>
#include <algorithm>
#include <iostream>

using namespace Optimal_Controller;

FuzzyLogic::FuzzyLogic(double min, double max)
  : min(min)
  , max(max)
  , controlInput(0)
  , error(0)
  , errorDiff(0)
{
}

FuzzyLogic::~FuzzyLogic()
{
}

double FuzzyLogic::getControlInput(const double setpoint, const double measurement, double dt)
{
  static double maxSetPoint = std::max(maxSetPoint, setpoint);
  static double maxMeasurement = std::max(maxMeasurement, measurement);
  static double minSetPoint = std::min(minSetPoint, setpoint);
  static double minMeasurement = std::min(minMeasurement, measurement);

  double maxError = maxSetPoint - maxMeasurement;
  double minError = minSetPoint - minMeasurement;
  
  error = normalize((setpoint - measurement), maxError, minError);
  errorDiff = error / dt;

  if (error == 0.0 && errorDiff == 0.0)
  {
    return controlInput;
  }

  controlInput = fuzzyRule();

  double ret = denormalize(controlInput, maxError, minError);

  // Hard constraints
  if (ret < min)
  {
    ret = min;
  }
  else if (ret > max)
  {
    ret = max;
  }

  return ret;
}

double FuzzyLogic::low(double x)
{
  return std::max(0.0, std::min(1.0, (x + 1) / 2));
}

double FuzzyLogic::high(double x)
{
  return std::max(0.0, std::min(1.0, (x - 1) / -2));
}

double FuzzyLogic::negLow(double x)
{
  return std::max(0.0, std::min(1.0, (-x + 1) / 2));
}

double FuzzyLogic::negHigh(double x)
{
  return std::max(0.0, std::min(1.0, (-x - 1) / -2));
}

double FuzzyLogic::normalize(double error, double maxError, double minError)
{
  return error != 0 ? (error - minError) / (maxError - minError) : 0;
}

double FuzzyLogic::denormalize(double controlInput, double maxError, double minError)
{
  return (controlInput * (maxError - minError)) + minError;
}

double FuzzyLogic::fuzzyRule()
{
  controlInput = rule1();
  controlInput = rule2();
  controlInput = rule3();
  controlInput = rule4();

  return controlInput;
}

double FuzzyLogic::rule1()
{
  controlInput = std::max(controlInput, low(error) * low(errorDiff) * high(0));
  controlInput = std::max(controlInput, negLow(error) * low(errorDiff) * high(0));

  return controlInput;
}

double FuzzyLogic::rule2()
{
  controlInput = std::max(controlInput, low(error) * high(errorDiff) * high(0));
  controlInput = std::max(controlInput, negLow(error) * high(errorDiff) * high(0));

  return controlInput;
}

double FuzzyLogic::rule3()
{
  controlInput = std::max(controlInput, high(error) * low(errorDiff) * high(0));
  controlInput = std::max(controlInput, negHigh(error) * low(errorDiff) * high(0));

  return controlInput;
}

double FuzzyLogic::rule4()
{
  controlInput = std::max(controlInput, high(error) * high(errorDiff) * high(0));
  controlInput = std::max(controlInput, negHigh(error) * high(errorDiff) * high(0));

  return controlInput;
}
