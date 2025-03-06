#include "matrix.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <cassert>
#include <iomanip>

// Constructor for empty matrix
Matrix::Matrix() : rows(0), cols(0) {}

// Constructor for matrix of specified dimensions
Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data.resize(rows * cols, 0.0);
}

// Constructor for matrix with initial data
Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& initialData) : rows(rows), cols(cols) {
    if (initialData.size() != rows * cols) {
        throw std::invalid_argument("Initial data size doesn't match matrix dimensions");
    }
    data = initialData;
}

// Copy constructor
Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}

// Move constructor
Matrix::Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
    other.rows = 0;
    other.cols = 0;
}

// Copy assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

// Move assignment operator
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = std::move(other.data);
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

// Element access
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[row * cols + col];
}

// Const element access
const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[row * cols + col];
}

// Addition
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

// Subtraction
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions don't match for subtraction");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// Matrix multiplication
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Scalar multiplication
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

// Transpose
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Hadamard (element-wise) product
Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

// Apply function element-wise
Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = func(data[i]);
    }
    return result;
}

// Initialize with random values
void Matrix::randomize(double mean, double stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
}

// Xavier/Glorot initialization
void Matrix::initializeXavier(size_t inputSize) {
    double limit = std::sqrt(6.0 / (inputSize + cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
}

// He initialization
void Matrix::initializeHe(size_t inputSize) {
    double stddev = std::sqrt(2.0 / inputSize);
    randomize(0.0, stddev);
}

// Convert matrix to vector (flatten)
std::vector<double> Matrix::toVector() const {
    return data;
}

// Create a matrix from a vector
Matrix Matrix::fromVector(const std::vector<double>& vec, bool columnVector) {
    if (columnVector) {
        Matrix result(vec.size(), 1);
        result.data = vec;
        return result;
    } else {
        Matrix result(1, vec.size());
        result.data = vec;
        return result;
    }
}

// Get a specific row as a vector
std::vector<double> Matrix::getRow(size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Row index out of range");
    }
    
    std::vector<double> rowVec(cols);
    for (size_t j = 0; j < cols; ++j) {
        rowVec[j] = (*this)(row, j);
    }
    return rowVec;
}

// Get a specific column as a vector
std::vector<double> Matrix::getColumn(size_t col) const {
    if (col >= cols) {
        throw std::out_of_range("Column index out of range");
    }
    
    std::vector<double> colVec(rows);
    for (size_t i = 0; i < rows; ++i) {
        colVec[i] = (*this)(i, col);
    }
    return colVec;
}

// Set all elements to a specific value
void Matrix::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

// Get dimensions
size_t Matrix::getRows() const {
    return rows;
}

size_t Matrix::getCols() const {
    return cols;
}

// Reshape matrix
void Matrix::reshape(size_t newRows, size_t newCols) {
    if (newRows * newCols != rows * cols) {
        throw std::invalid_argument("Cannot reshape: new dimensions don't match total size");
    }
    
    rows = newRows;
    cols = newCols;
}

// Print matrix for debugging
void Matrix::print(std::ostream& os) const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            os << std::setw(10) << std::fixed << std::setprecision(4) << (*this)(i, j);
        }
        os << std::endl;
    }
}

// Matrix dot product with a vector
std::vector<double> Matrix::dot(const std::vector<double>& vec) const {
    if (cols != vec.size()) {
        throw std::invalid_argument("Vector size doesn't match matrix columns");
    }
    
    std::vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += (*this)(i, j) * vec[j];
        }
    }
    return result;
}

// Calculate the sum of all elements
double Matrix::sum() const {
    return std::accumulate(data.begin(), data.end(), 0.0);
}

// Calculate the mean of all elements
double Matrix::mean() const {
    return sum() / data.size();
}

// Return identity matrix
Matrix Matrix::identity(size_t size) {
    Matrix result(size, size);
    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}