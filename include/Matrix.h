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
      Matrix(size_t rows, size_t cols);                     // Constructor
      size_t rows() const;                                  // Get number of rows
      size_t cols() const;                                  // Get number of columns
      void fill(const T& value);                            // Fill matrix with value
      void print() const;                                   // Print matrix
      T operator*(const Matrix<T>& other) const;            // Dot product if both are vectors
      T& operator()(size_t i, size_t j);                    // Non-const element access
      T operator()(size_t i, size_t j) const;               // Const element access
      Matrix& operator<<(T values);                         // Chaining matrix initialization
      Matrix& operator,(T values);                          // Allow chaining of << with comma
      Matrix<T> transpose() const;                          // Transpose Matrix
      Matrix<T> inverse() const;                            // Inverse of Matrix

    private:
      size_t m_rows;              // Number of rows
      size_t m_cols;              // Number of columns
      size_t m_currentElement;    // Track the current element for << operator
      std::vector<T> m_data;      // Matrix data in a flat array

      void check_indices(size_t i, size_t j) const;  // Check indices bounds
      bool is_vector() const;                        // Check if Matrix is 1xN or Nx1
      bool is_square() const;                        // Check if the matrix is square
  };

  // Implementation of Matrix methods

  template <typename T>
  Matrix<T>::Matrix(size_t i_rows, size_t i_cols)
    : m_rows(i_rows)
    , m_cols(i_cols)
    , m_currentElement(0)
    , m_data(i_rows * i_cols)
  {
    if (i_rows == 0 || i_cols == 0)
    {
      throw std::invalid_argument("Number of i_rows and i_cols must be greater than zero.");
    }
  }

  template <typename T>
  size_t Matrix<T>::rows() const
  {
    return m_rows;
  }

  template <typename T>
  size_t Matrix<T>::cols() const
  {
    return m_cols;
  }

  template <typename T>
  void Matrix<T>::fill(const T& i_value)
  {
    std::fill(m_data.begin(), m_data.end(), i_value);
  }

  template <typename T>
  void Matrix<T>::print() const
  {
    for (size_t i = 0; i < m_rows; ++i)
    {
      for (size_t j = 0; j < m_cols; ++j)
      {
        std::cout << (*this)(i, j) << ' ';
      }
      std::cout << '\n';
    }
  }

  template <typename T>
  T Matrix<T>::operator*(const Matrix<T>& i_other) const
  {
    // Ensure both matrices are vectors and have the same number of elements
    if (!is_vector() || !i_other.is_vector() || (m_rows * m_cols != i_other.m_rows * i_other.m_cols))
    {
      throw std::invalid_argument("Both matrices must be vectors of the same size for dot product.");
    }

    T w_result = T{};
    for (size_t i = 0; i < m_rows * m_cols; ++i)
    {
      w_result += m_data[i] * i_other.m_data[i];
    }

    return w_result;
  }

  template <typename T>
  T& Matrix<T>::operator()(size_t i, size_t j)
  {
    check_indices(i, j);
    return m_data[i * m_cols + j];
  }

  template <typename T>
  T Matrix<T>::operator()(size_t i, size_t j) const
  {
    check_indices(i, j);
    return m_data[i * m_cols + j];
  }

  template <typename T>
  Matrix<T>& Matrix<T>::operator<<(T i_values) 
  {
    if (m_currentElement >= m_rows * m_cols)
    {
      std::cout << m_currentElement << "\t" << m_rows << "\t" << m_cols << std::endl;
      throw std::out_of_range("Matrix<T>::operator<<(...)\nToo many elements provided for matrix initialization");
    }
    m_data[m_currentElement++] = i_values;
    return *this;
  }

  template <typename T>
  Matrix<T>& Matrix<T>::operator,(T i_value)
  {
    return this->operator<<(i_value);
  }

  template <typename T>
  Matrix<T> Matrix<T>::transpose() const
  {
    Matrix<T> w_transposed(m_cols, m_rows); // Create a new Matrix with swapped dimensions
    for (size_t i = 0; i < m_rows; ++i)
    {
      for (size_t j = 0; j < m_cols; ++j)
      {
        w_transposed(j, i) = (*this)(i, j); // Set w_transposed elements
      }
    }
    return w_transposed; // Return the w_transposed matrix
  }

  template <typename T>
  bool Matrix<T>::is_square() const
  {
    return m_rows == m_cols;
  }

  template <typename T>
  Matrix<T> Matrix<T>::inverse() const
  {
    if (!is_square())
    {
      throw std::invalid_argument("Matrix must be square to compute its inverse.");
    }

    size_t w_n = m_rows;
    Matrix<T> w_augmented(w_n, 2 * w_n);

    // Create an w_augmented matrix [A | I]
    for (size_t i = 0; i < w_n; ++i)
    {
      for (size_t j = 0; j < w_n; ++j)
      {
        w_augmented(i, j) = (*this)(i, j); // Copy the original matrix
      }
      for (size_t j = w_n; j < 2 * w_n; ++j)
      {
        w_augmented(i, j) = (j == i + w_n) ? static_cast<T>(1) : static_cast<T>(0); // Identity matrix
      }
    }

    // Perform Gaussian elimination
    for (size_t i = 0; i < w_n; ++i)
    {
      // Find the maximum element in the current column for pivoting
      size_t w_maxRow = i;
      for (size_t k = i + 1; k < w_n; ++k)
      {
        if (std::abs(w_augmented(k, i)) > std::abs(w_augmented(w_maxRow, i)))
        {
          w_maxRow = k;
        }
      }

      // Swap maximum row with current row
      if (w_maxRow != i)
      {
        for (size_t k = 0; k < 2 * w_n; ++k)
        {
          std::swap(w_augmented(w_maxRow, k), w_augmented(i, k));
        }
      }

      // Check if the pivot element is zero (singular matrix)
      T w_pivot = w_augmented(i, i);
      if (w_pivot == 0)
      {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
      }

      // Normalize the pivot row
      for (size_t j = 0; j < 2 * w_n; ++j)
      {
        w_augmented(i, j) /= w_pivot; // Scale row
      }

      // Eliminate the current column in the rows below
      for (size_t j = i + 1; j < w_n; ++j)
      {
        T w_factor = w_augmented(j, i);
        for (size_t k = 0; k < 2 * w_n; ++k)
        {
          w_augmented(j, k) -= w_factor * w_augmented(i, k);
        }
      }
    }

    // Back substitution to eliminate entries above the pivot
    for (size_t i = w_n; i-- > 0;)
    {
      for (size_t j = 0; j < i; ++j) // Eliminate above
      {
        T w_factor = w_augmented(j, i);
        for (size_t k = 0; k < 2 * w_n; ++k)
        {
          w_augmented(j, k) -= w_factor * w_augmented(i, k);
        }
      }
    }

    // Extract the right half as the inverse
    Matrix<T> w_inv(w_n, w_n);
    for (size_t i = 0; i < w_n; ++i)
    {
      for (size_t j = 0; j < w_n; ++j)
      {
        w_inv(i, j) = w_augmented(i, j + w_n); // Copy the right side
      }
    }

    return w_inv; // Return the inverse matrix
  }

  template <typename T>
  void Matrix<T>::check_indices(size_t w_i, size_t w_j) const
  {
    if (w_i >= m_rows || w_j >= m_cols)
    {
      throw std::out_of_range("Matrix indices are out of bounds.");
    }
  }

  template <typename T>
  bool Matrix<T>::is_vector() const
  {
    return m_rows == 1 || m_cols == 1;
  }
}

#endif /* MATRIX_H */
