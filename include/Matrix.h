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
#include <cmath>
#include <complex>

namespace Optimal_Controller
{
  template <typename T>
  class Matrix
  {
    public:
      /**
       * @brief Constructor
       * 
       * @param i_rows 
       * @param i_cols 
       */
      Matrix(size_t i_rows, size_t i_cols)
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

      /**
       * @brief Get number of rows
       * 
       * @return size_t 
       */
      size_t rows() const
      {
        return m_rows;
      }

      /**
       * @brief Get number of columns
       * 
       * @return size_t 
       */
      size_t cols() const
      {
        return m_cols;
      }

      /**
       * @brief Fill matrix with a value
       * 
       * @param i_value 
       */
      void fill(const T& i_value)
      {
        std::fill(m_data.begin(), m_data.end(), i_value);
      }

      /**
       * @brief Function to zero out the matrix
       * 
       */
      void zero()
      {
        std::fill(m_data.begin(), m_data.end(), 0);
        m_currentElement = 0; // Reset current element index if necessary
      }

      /**
       * @brief Print matrix
       * 
       */
      void print() const
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

      /**
       * @brief Block from the matrix
       * 
       * @param w_row_start 
       * @param w_col_start 
       * @param w_row_size 
       * @param w_col_size 
       * @return Matrix<T> 
       */
      Matrix<T> block(size_t w_row_start, size_t w_col_start, size_t w_row_size, size_t w_col_size) const
      {
        // Check if the block dimensions are valid
        if (w_row_start + w_row_size > m_rows || w_col_start + w_col_size > m_cols)
        {
          throw std::out_of_range("Requested block is out of matrix bounds.");
        }

        // Create a matrix to store the block
        Matrix<T> w_result(w_row_size, w_col_size);

        // Fill the block matrix with appropriate values
        for (size_t i = 0; i < w_row_size; ++i)
        {
          for (size_t j = 0; j < w_col_size; ++j)
          {
            w_result(i, j) = (*this)(w_row_start + i, w_col_start + j);
          }
        }

        return w_result;
      }

      /**
       * @brief Matrix Multiplication
       * 
       * @param i_other 
       * @return T 
       */
      T operator*(const Matrix<T>& i_other) const
      {
        if (m_cols != i_other.m_rows)
        {
          throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix<T> w_result(m_rows, i_other.m_cols);

        for (size_t i = 0; i < m_rows; ++i)
        {
          for (size_t j = 0; j < i_other.m_cols; ++j)
          {
            for (size_t k = 0; k < m_cols; ++k)
            {
              w_result(i, j) += (*this)(i, k) * i_other(k, j);
            }
          }
        }

        return w_result;
      }

      /**
       * @brief Non-const element access
       * 
       * @param i 
       * @param j 
       * @return T& 
       */
      T& operator()(size_t i, size_t j)
      {
        check_indices(i, j);
        return m_data[i * m_cols + j];
      }

      /**
       * @brief Const element access
       * 
       * @param i 
       * @param j 
       * @return T 
       */
      T operator()(size_t i, size_t j) const
      {
        check_indices(i, j);
        return m_data[i * m_cols + j];
      }

      /**
       * @brief Chaining matrix initialization
       * 
       * @param i_values 
       * @return Matrix& 
       */
      Matrix& operator<<(T i_values) 
      {
        if (m_currentElement >= m_rows * m_cols)
        {
          std::cout << m_currentElement << "\t" << m_rows << "\t" << m_cols << std::endl;
          throw std::out_of_range("Matrix<T>::operator<<(...)\nToo many elements provided for matrix initialization");
        }
        m_data[m_currentElement++] = i_values;
        return *this;
      }

      /**
       * @brief Allow chaining of << with comma
       * 
       * @param i_value 
       * @return Matrix& 
       */
      Matrix& operator,(T i_value)
      {
        return this->operator<<(i_value);
      }

      /**
       * @brief Transpose Matrix
       * 
       * @return Matrix<T> 
       */
      Matrix<T> transpose() const
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

      /**
       * @brief Inverse of Matrix
       * 
       * @return Matrix<T> 
       */
      Matrix<T> inverse() const
      {
        if (!is_square())
        {
          throw std::invalid_argument("Matrix must be square to compute its inverse.");
        }

        size_t n = m_rows;
        Matrix<T> w_augmented(n, 2 * n);

        // Create an w_augmented matrix [A | I]
        for (size_t i = 0; i < n; ++i)
        {
          for (size_t j = 0; j < n; ++j)
          {
            w_augmented(i, j) = (*this)(i, j); // Copy the original matrix
          }
          for (size_t j = n; j < 2 * n; ++j)
          {
            w_augmented(i, j) = (j == i + n) ? static_cast<T>(1) : static_cast<T>(0); // Identity matrix
          }
        }

        // Perform Gaussian elimination
        for (size_t i = 0; i < n; ++i)
        {
          // Find the maximum element in the current column for pivoting
          size_t w_maxRow = i;
          for (size_t k = i + 1; k < n; ++k)
          {
            if (std::abs(w_augmented(k, i)) > std::abs(w_augmented(w_maxRow, i)))
            {
              w_maxRow = k;
            }
          }

          // Swap maximum row with current row
          if (w_maxRow != i)
          {
            for (size_t k = 0; k < 2 * n; ++k)
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
          for (size_t j = 0; j < 2 * n; ++j)
          {
            w_augmented(i, j) /= w_pivot; // Scale row
          }

          // Eliminate the current column in the rows below
          for (size_t j = i + 1; j < n; ++j)
          {
            T w_factor = w_augmented(j, i);
            for (size_t k = 0; k < 2 * n; ++k)
            {
              w_augmented(j, k) -= w_factor * w_augmented(i, k);
            }
          }
        }

        // Back substitution to eliminate entries above the pivot
        for (size_t i = n; i-- > 0;)
        {
          for (size_t j = 0; j < i; ++j) // Eliminate above
          {
            T w_factor = w_augmented(j, i);
            for (size_t k = 0; k < 2 * n; ++k)
            {
              w_augmented(j, k) -= w_factor * w_augmented(i, k);
            }
          }
        }

        // Extract the right half as the inverse
        Matrix<T> w_inv(n, n);
        for (size_t i = 0; i < n; ++i)
        {
          for (size_t j = 0; j < n; ++j)
          {
            w_inv(i, j) = w_augmented(i, j + n); // Copy the right side
          }
        }

        return w_inv; // Return the inverse matrix
      }

      /**
       * @brief Get the Eigenvalues of Matrix
       * 
       * @return std::vector<std::complex<T>> for ::real and ::imag extraction
       */
      std::vector<std::complex<T>> eigenvalues()
      {
        if (m_rows != m_cols)
        {
          throw std::invalid_argument("This method only supports square matrices.");
        }

        size_t n = m_rows;
        Matrix<T> A = *this; // Initialize the matrix for QR iteration
        std::vector<std::complex<T>> w_eigenvalues(n); // Eigenvalue storage

        const size_t w_max_iterations = 1000; // Limit iterations
        const T w_tolerance = 1e-10; // Convergence tolerance

        for (size_t iter = 0; iter < w_max_iterations; ++iter)
        {
          // Perform QR decomposition
          Matrix<T> Q(n, n), R(n, n);
          qrDecomposition(A, Q, R);

          A = R * Q; // Reconstruct A from R and Q

          // Check for convergence
          T w_offDiagonalSum = 0;
          for (size_t i = 0; i < n - 1; ++i)
          {
            w_offDiagonalSum += std::abs(A(i, i + 1));
          }

          if (w_offDiagonalSum < w_tolerance)
          {
            break; // Convergence achieved
          }
        }

        // Extract eigenvalues from the diagonal
        for (size_t i = 0; i < n; ++i)
        {
          w_eigenvalues[i] = A(i, i);
        }

        return w_eigenvalues;
      }

      /**
       * @brief Get the Eigenvector of Matrix
       * 
       * @return Matrix<T> 
       * @param i_eigenvalue
       */
      Matrix<T> eigenvectors(T i_eigenvalue) const
      {
        if (m_rows != m_cols)
        {
          throw std::invalid_argument("This method only supports square matrices.");
        }

        size_t n = m_rows;
        Matrix<T> w_eigenvector(n, 1);        // Initialize an n x 1 eigenvector
        Matrix<std::complex<T>> w_temp(n, n); // Create a temporary matrix for A - λI

        for (size_t i = 0; i < n; ++i)
        {
          for (size_t j = 0; j < n; ++j)
          {
            w_temp(i, j) = (*this)(i, j) - (i == j ? i_eigenvalue : static_cast<T>(0)); // A - λI
          }
        }

        // Solve the system of equations (A - λI)x = 0 using Gaussian elimination with partial pivoting
        if (gaussianElimination(w_temp, w_eigenvector) == 0)
        {
          // Normalize the eigenvector
          normalizeEigenvector(w_eigenvector);
          return w_eigenvector;
        }
        else
        {
          throw std::runtime_error("No valid eigenvector found for the given eigenvalue.");
        }
      }

    private:
      size_t m_rows;              // Number of rows
      size_t m_cols;              // Number of columns
      size_t m_currentElement;    // Track the current element for << operator
      std::vector<T> m_data;      // Matrix data in a flat array

      /**
       * @brief Check indices bounds
       * 
       * @param i 
       * @param j 
       */
      void check_indices(size_t i, size_t j) const
      {
        if (i >= m_rows || j >= m_cols)
        {
          throw std::out_of_range("Matrix indices are out of bounds.");
        }
      }

      /**
       * @brief Check if Matrix is 1xN or Nx1
       * 
       * @return true 
       * @return false 
       */
      bool is_vector() const
      {
        return m_rows == 1 || m_cols == 1;
      }
      
      /**
       * @brief Check if the matrix is square
       * 
       * @return true 
       * @return false 
       */
      bool is_square() const
      {
        return m_rows == m_cols;
      }

      /**
       * @brief QR decomposition (uses Gram-Schmidt process)
       * 
       * @param A 
       * @param Q 
       * @param R 
       */
      void qrDecomposition(const Matrix<T>& A, Matrix<T>& Q, Matrix<T>& R)
      {
        size_t n = A.m_rows;
        Q = Matrix<T>(n, n);
        R = Matrix<T>(n, n);

        for (size_t j = 0; j < n; ++j)
        {
          // Copy column j of A to v
          for (size_t i = 0; i < n; ++i)
          {
            Q(i, j) = A(i, j);
          }

          // Orthogonalization
          for (size_t i = 0; i < j; ++i)
          {
            T dotProduct = 0;
            for (size_t k = 0; k < n; ++k)
            {
              dotProduct += Q(k, i) * Q(k, j);
            }

            R(i, j) = dotProduct;

            for (size_t k = 0; k < n; ++k)
            {
              Q(k, j) -= dotProduct * Q(k, i);
            }
          }

          // Normalization
          T norm = 0;
          for (size_t i = 0; i < n; ++i)
          {
            norm += Q(i, j) * Q(i, j);
          }

          norm = std::sqrt(norm);

          R(j, j) = norm;

          // Avoid division by zero
          if (norm > 1e-10)
          { 
            for (size_t i = 0; i < n; ++i)
            {
              Q(i, j) /= norm;
            }
          }
        }
      }

      /**
       * @brief Normalize Eigenvector of Matrix
       * 
       * @param w_eigenvector 
       */
      void normalizeEigenvector(Matrix<T>& w_eigenvector) const
      {
        T norm = 0;
        for (size_t i = 0; i < w_eigenvector.m_rows; ++i)
        {
          norm += w_eigenvector(i, 0) * w_eigenvector(i, 0);
        }
        norm = std::sqrt(norm);
        if (norm > 0)
        {
          for (size_t i = 0; i < w_eigenvector.m_rows; ++i)
          {
            w_eigenvector(i, 0) /= norm; // Normalize the eigenvector
          }
        }
      }

      /**
       * @brief Gaussian Elimination 
       * 
       * @param w_aug 
       * @param w_eigenvector 
       * @return int 
       */
      int gaussianElimination(Matrix<T>& w_aug, Matrix<T>& w_eigenvector) const
      {
        size_t n = w_aug.m_rows;

        // Perform Gaussian elimination with partial pivoting
        for (size_t i = 0; i < n; ++i)
        {
          // Partial pivoting
          size_t max_row = i;
          for (size_t k = i + 1; k < n; ++k)
          {
            if (std::abs(w_aug(k, i)) > std::abs(w_aug(max_row, i)))
            {
              max_row = k;
            }
          }

          std::swap(w_aug(i), w_aug(max_row)); // Swap the current row with the max row

          // Make the diagonal contain all 1s
          T w_diag = w_aug(i, i);

          if (std::abs(w_diag) < 1e-12)
          {
            return 1; // Singular matrix; no unique solution
          }
          for (size_t j = 0; j < n; ++j)
          {
            w_aug(i, j) /= w_diag;
          }

          // Eliminate the column below the pivot
          for (size_t j = i + 1; j < n; ++j)
          {
            T factor = w_aug(j, i);

            for (size_t k = 0; k < n; ++k)
            {
              w_aug(j, k) -= factor * w_aug(i, k);
            }
          }
        }

        // Now back substitute to find the eigenvector
        for (size_t i = n - 1; i != SIZE_MAX; --i)
        {
          bool non_zero_row = false;
          for (size_t j = 0; j < n; ++j)
          {
            if (w_aug(i, j) != 0)
            {
              non_zero_row = true;
              break;
            }
          }
          
          if (non_zero_row)
          {
            w_eigenvector(i, 0) = 1; // Set a basic solution
            
            for (size_t j = i + 1; j < n; ++j)
            {
              w_eigenvector(i, 0) -= w_aug(i, j) * w_eigenvector(j, 0);
            }

            break; // Exit after finding the first non-zero row
          }
        }

        return 0; // Indicate success
      }
  };
}

#endif /* MATRIX_H */
