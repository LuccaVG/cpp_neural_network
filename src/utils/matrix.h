#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>
#include <iostream>

/**
 * @brief Matrix class for neural network computations
 */
class Matrix {
private:
    size_t rows;
    size_t cols;
    std::vector<double> data;

public:
    // Constructors and assignment operators
    Matrix();  // Default constructor
    Matrix(size_t rows, size_t cols);  // Size constructor
    Matrix(size_t rows, size_t cols, const std::vector<double>& initialData);  // Data constructor
    Matrix(const Matrix& other);  // Copy constructor
    Matrix(Matrix&& other) noexcept;  // Move constructor
    Matrix& operator=(const Matrix& other);  // Copy assignment
    Matrix& operator=(Matrix&& other) noexcept;  // Move assignment

    // Element access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    // Matrix operations
    Matrix operator+(const Matrix& other) const;  // Addition
    Matrix operator-(const Matrix& other) const;  // Subtraction
    Matrix operator*(const Matrix& other) const;  // Matrix multiplication
    Matrix operator*(double scalar) const;  // Scalar multiplication
    Matrix transpose() const;  // Transpose
    Matrix hadamard(const Matrix& other) const;  // Element-wise multiplication
    Matrix apply(std::function<double(double)> func) const;  // Apply function element-wise

    // Initialization methods
    void randomize(double mean = 0.0, double stddev = 1.0);  // Random initialization
    void initializeXavier(size_t inputSize);  // Xavier/Glorot initialization
    void initializeHe(size_t inputSize);  // He initialization
    void fill(double value);  // Fill with constant

    // Conversion methods
    std::vector<double> toVector() const;  // Convert to vector
    static Matrix fromVector(const std::vector<double>& vec, bool columnVector = true);  // Create from vector

    // Access methods
    std::vector<double> getRow(size_t row) const;  // Get specific row
    std::vector<double> getColumn(size_t col) const;  // Get specific column
    size_t getRows() const;  // Get number of rows
    size_t getCols() const;  // Get number of columns

    // Utility methods
    void reshape(size_t newRows, size_t newCols);  // Reshape matrix dimensions
    void print(std::ostream& os = std::cout) const;  // Print matrix
    std::vector<double> dot(const std::vector<double>& vec) const;  // Matrix-vector product
    double sum() const;  // Sum of all elements
    double mean() const;  // Mean of all elements
    static Matrix identity(size_t size);  // Create identity matrix
};

#endif // MATRIX_H