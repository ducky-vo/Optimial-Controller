/**
 * @file FuzzyLogic.h
 * @author lsvng
 * @brief Fuzzy logic controller
 * 
 */

#ifndef FUZZYLOGIC_H
#define FUZZYLOGIC_H

namespace Optimal_Controller
{
  class FuzzyLogic
  {
    public:
      /**
       * @brief Construct a new Fuzzy Logic object
       * 
       * @param min minimum value of manipulated variable
       * @param max maximum value of manipulated variable
       */
      FuzzyLogic(double min, double max);

      /**
       * @brief Destroy the Fuzzy Logic object
       * 
       */
      ~FuzzyLogic();

      /**
       * @brief Disable copy constructor
       * 
       * @param iFuzzyLogic 
       */
      FuzzyLogic(FuzzyLogic& iFuzzyLogic) = delete;

      /**
       * @brief Disable assignment constructor
       * 
       * @param iFuzzyLogic 
       */
      void operator=(FuzzyLogic& iFuzzyLogic) = delete;

      /**
       * @brief Get the Control Input
       * 
       * @param setpoint 
       * @param measurement 
       * @param dt 
       * @return double 
       */
      double getControlInput(const double setpoint, const double measurement, double dt);

    private:
      /**
       * @brief Compute low state
       * 
       * @param x error || errorDiff
       * @return double 
       */
      double low(double x);

      /**
       * @brief Compute high state
       * 
       * @param x error || errorDiff
       * @return double 
       */
      double high(double x);

      /**
       * @brief Compute negive low state
       * 
       * @param x error || errorDiff
       * @return double 
       */
      double negLow(double x);

      /**
       * @brief Compute negative high state
       * 
       * @param x error || errorDiff
       * @return double 
       */
      double negHigh(double x);

      /**
       * @brief Normalizing input data
       * 
       * @param error 
       * @param maxError 
       * @param minError 
       * @return double 
       */
      double normalize(double error, double maxError, double minError);

      /**
       * @brief Denormalizing output data
       * 
       * @param output 
       * @param maxError 
       * @param minError 
       * @return double 
       */
      double denormalize(double output, double maxError, double minError);

      /**
       * @brief Run all 4 rules of a logic controller
       * 
       * @return double 
       */
      double fuzzyRule();

      /**
       * @brief If error is low and error differential is low, then output is high
       * 
       * @return double 
       */
      double rule1();

      /**
       * @brief If error is low and error differential is high, then output is zero
       * 
       * @return double 
       */
      double rule2();

      /**
       * @brief If error is high and error differential is low, then output is zero
       * 
       * @return double 
       */
      double rule3();

      /**
       * @brief If error is high and error differential is high, then output is zero
       * 
       * @return double 
       */
      double rule4();

    private:
      double min;          // minimum value of manipulated variable
      double max;          // maximum value of manipulated variable
      double controlInput; // Control input
      double error;        // State error
      double errorDiff;    // Differential state error
  };
}

#endif /* FUZZYLOGIC_H */
