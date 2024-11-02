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
      Matrix<T> transpose() const;                // Transpose Matrix
      Matrix<T> inverse() const;                  // Inverse of Matrix

    private:
      size_t mRows;              // Number of rows
      size_t mCols;              // Number of columns
      std::vector<T> mData;      // Matrix data in a flat array

      void check_indices(size_t i, size_t j) const;  // Check indices bounds
      bool is_vector() const;                        // Check if Matrix is 1xN or Nx1
      bool is_square() const;                        // Check if the matrix is square
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
  Matrix<T> Matrix<T>::transpose() const
  {
    Matrix<T> transposed(mCols, mRows); // Create a new Matrix with swapped dimensions
    for (size_t i = 0; i < mRows; ++i)
    {
      for (size_t j = 0; j < mCols; ++j)
      {
        transposed(j, i) = (*this)(i, j); // Set transposed elements
      }
    }
    return transposed; // Return the transposed matrix
  }

  template <typename T>
  bool Matrix<T>::is_square() const
  {
    return mRows == mCols;
  }

  template <typename T>
  Matrix<T> Matrix<T>::inverse() const
  {
    if (!is_square())
    {
      throw std::invalid_argument("Matrix must be square to compute its inverse.");
    }

    size_t n = mRows;
    Matrix<T> augmented(n, 2 * n);

    // Create an augmented matrix [A | I]
    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < n; ++j)
      {
        augmented(i, j) = (*this)(i, j); // Copy the original matrix
      }
      for (size_t j = n; j < 2 * n; ++j)
      {
        augmented(i, j) = (j == i + n) ? static_cast<T>(1) : static_cast<T>(0); // Identity matrix
      }
    }

    // Perform Gaussian elimination
    for (size_t i = 0; i < n; ++i)
    {
      // Find the maximum element in the current column for pivoting
      size_t maxRow = i;
      for (size_t k = i + 1; k < n; ++k)
      {
        if (std::abs(augmented(k, i)) > std::abs(augmented(maxRow, i)))
        {
          maxRow = k;
        }
      }

      // Swap maximum row with current row
      if (maxRow != i)
      {
        for (size_t k = 0; k < 2 * n; ++k)
        {
          std::swap(augmented(maxRow, k), augmented(i, k));
        }
      }

      // Check if the pivot element is zero (singular matrix)
      T pivot = augmented(i, i);
      if (pivot == 0)
      {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
      }

      // Normalize the pivot row
      for (size_t j = 0; j < 2 * n; ++j)
      {
        augmented(i, j) /= pivot; // Scale row
      }

      // Eliminate the current column in the rows below
      for (size_t j = i + 1; j < n; ++j)
      {
        T factor = augmented(j, i);
        for (size_t k = 0; k < 2 * n; ++k)
        {
          augmented(j, k) -= factor * augmented(i, k);
        }
      }
    }

    // Back substitution to eliminate entries above the pivot
    for (size_t i = n; i-- > 0;)
    {
      for (size_t j = 0; j < i; ++j) // Eliminate above
      {
        T factor = augmented(j, i);
        for (size_t k = 0; k < 2 * n; ++k)
        {
          augmented(j, k) -= factor * augmented(i, k);
        }
      }
    }

    // Extract the right half as the inverse
    Matrix<T> inv(n, n);
    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < n; ++j)
      {
        inv(i, j) = augmented(i, j + n); // Copy the right side
      }
    }

    return inv; // Return the inverse matrix
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
