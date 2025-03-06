#include "matrix.h"
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <vector>

// Function to multiply two matrices
std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    size_t colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// Function to add two matrices
std::vector<std::vector<double>> matrixAdd(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    size_t colsB = B[0].size();

    if (rowsA != rowsB || colsA != colsB) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsA, 0.0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsA; ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}

// Function to transpose a matrix
std::vector<std::vector<double>> matrixTranspose(const std::vector<std::vector<double>>& A) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();

    std::vector<std::vector<double>> result(colsA, std::vector<double>(rowsA, 0.0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsA; ++j) {
            result[j][i] = A[i][j];
        }
    }

    return result;
}

// Function to apply element-wise activation function
std::vector<std::vector<double>> matrixApplyActivation(const std::vector<std::vector<double>>& A, std::function<double(double)> activationFunc) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsA, 0.0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsA; ++j) {
            result[i][j] = activationFunc(A[i][j]);
        }
    }

    return result;
}