#include "activation.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

double Activation::apply(double x, ActivationType type) {
    switch (type) {
        case ActivationType::LINEAR:
            return x;
        case ActivationType::SIGMOID:
            return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::TANH:
            return std::tanh(x);
        case ActivationType::RELU:
            return std::max(0.0, x);
        case ActivationType::LEAKY_RELU:
            return x > 0.0 ? x : 0.01 * x;
        case ActivationType::ELU:
            return x > 0.0 ? x : std::exp(x) - 1.0;
        case ActivationType::SWISH:
            return x * apply(x, ActivationType::SIGMOID);
        default:
            throw std::runtime_error("Unsupported activation function");
    }
}

std::vector<double> Activation::apply(const std::vector<double>& x, ActivationType type) {
    std::vector<double> result(x.size());
    
    if (type == ActivationType::SOFTMAX) {
        double max_val = *std::max_element(x.begin(), x.end());
        double sum = 0.0;
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] /= sum;
        }
    } else {
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = apply(x[i], type);
        }
    }
    
    return result;
}

double Activation::derivative(double x, ActivationType type) {
    switch (type) {
        case ActivationType::LINEAR:
            return 1.0;
        case ActivationType::SIGMOID: {
            double sigmoid = apply(x, ActivationType::SIGMOID);
            return sigmoid * (1.0 - sigmoid);
        }
        case ActivationType::TANH: {
            double tanh_x = std::tanh(x);
            return 1.0 - tanh_x * tanh_x;
        }
        case ActivationType::RELU:
            return x > 0.0 ? 1.0 : 0.0;
        case ActivationType::LEAKY_RELU:
            return x > 0.0 ? 1.0 : 0.01;
        case ActivationType::ELU:
            return x > 0.0 ? 1.0 : std::exp(x);
        case ActivationType::SWISH: {
            double sigmoid = apply(x, ActivationType::SIGMOID);
            return sigmoid + x * sigmoid * (1.0 - sigmoid);
        }
        default:
            throw std::runtime_error("Unsupported activation function derivative");
    }
}

std::vector<double> Activation::derivative(const std::vector<double>& x, ActivationType type) {
    std::vector<double> result(x.size());
    
    if (type == ActivationType::SOFTMAX) {
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = 1.0; // Simplified for softmax
        }
    } else {
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = derivative(x[i], type);
        }
    }
    
    return result;
}