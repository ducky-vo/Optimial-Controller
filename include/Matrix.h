/**
 * @file Matrix.h
 * @author lsvng
 * @brief Declaration and implementation of a templated Matrix class.
 * @tparam T The data type of the matrix elements.
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <stdexcept>

namespace Optimal_Controller
{
  template <typename T>
  class Matrix
  {
    public:
      Matrix(size_t rows, size_t cols);           // Constructor
      T& operator()(size_t i, size_t j);          // Non-const element access
      T operator()(size_t i, size_t j) const;     // Const element access
      size_t rows() const;                        // Get number of rows
      size_t cols() const;                        // Get number of columns
      void fill(const T& value);                  // Fill matrix with value
      void print() const;                         // Print matrix
      T operator*(const Matrix<T>& other) const;  // Overload * operator for dot product if both are vectors

    private:
      size_t mRows;              // Number of rows
      size_t mCols;              // Number of columns
      std::vector<T> mData;      // Matrix data in a flat array

      void check_indices(size_t i, size_t j) const;  // Check indices bounds
      bool is_vector() const;                        // Helper function to check if Matrix is 1xN or Nx1
  };

  // Implementation of Matrix methods

  template <typename T>
  Matrix<T>::Matrix(size_t rows, size_t cols)
    : mRows(rows), mCols(cols), mData(rows * cols)
  {
    if (rows == 0 || cols == 0)
    {
      throw std::invalid_argument("Number of rows and columns must be greater than zero.");
    }
  }

  template <typename T>
  T& Matrix<T>::operator()(size_t i, size_t j)
  {
    check_indices(i, j);
    return mData[i * mCols + j];
  }

  template <typename T>
  T Matrix<T>::operator()(size_t i, size_t j) const
  {
    check_indices(i, j);
    return mData[i * mCols + j];
  }

  template <typename T>
  size_t Matrix<T>::rows() const
  {
    return mRows;
  }

  template <typename T>
  size_t Matrix<T>::cols() const
  {
    return mCols;
  }

  template <typename T>
  void Matrix<T>::fill(const T& value)
  {
    std::fill(mData.begin(), mData.end(), value);
  }

  template <typename T>
  void Matrix<T>::print() const
  {
    for (size_t i = 0; i < mRows; ++i)
    {
      for (size_t j = 0; j < mCols; ++j)
      {
        std::cout << (*this)(i, j) << ' ';
      }
      
      std::cout << '\n';
    }
  }

  template <typename T>
  void Matrix<T>::check_indices(size_t i, size_t j) const
  {
    if (i >= mRows || j >= mCols)
    {
      throw std::out_of_range("Matrix indices are out of bounds.");
    }
  }

  template <typename T>
  bool Matrix<T>::is_vector() const
  {
    return mRows == 1 || mCols == 1;
  }

  template <typename T>
  T Matrix<T>::operator*(const Matrix<T>& other) const
  {
    // Ensure both matrices are vectors and have the same number of elements
    if (!is_vector() || !other.is_vector() || (mRows * mCols != other.mRows * other.mCols))
    {
      throw std::invalid_argument("Both matrices must be vectors of the same size for dot product.");
    }

    T result = T{};
    for (size_t i = 0; i < mRows * mCols; ++i)
    {
      result += mData[i] * other.mData[i];
    }

    return result;
  }
}

#endif /* MATRIX_H */
