// filepath: /cpp_neural_network/cpp_neural_network/src/utils/matrix.h

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>

class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    size_t getRows() const;
    size_t getCols() const;

    Matrix transpose() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;

private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;
};

#endif // MATRIX_H