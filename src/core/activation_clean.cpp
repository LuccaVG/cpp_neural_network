#include "activation.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Base class virtual method implementations
std::vector<double> Activation::activate(const std::vector<double>& inputs) const {
    std::vector<double> result(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        result[i] = activate(inputs[i]);
    }
    return result;
}

std::vector<double> Activation::derivative(const std::vector<double>& inputs) const {
    std::vector<double> result(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        result[i] = derivative(inputs[i]);
    }
    return result;
}

// Factory method
std::unique_ptr<Activation> Activation::create(ActivationType type, double param) {
    switch (type) {
        case ActivationType::LINEAR:
            return std::make_unique<Linear>();
        case ActivationType::SIGMOID:
            return std::make_unique<Sigmoid>();
        case ActivationType::TANH:
            return std::make_unique<Tanh>();
        case ActivationType::RELU:
            return std::make_unique<ReLU>();
        case ActivationType::LEAKY_RELU:
            return std::make_unique<LeakyReLU>(param > 0 ? param : 0.01);
        case ActivationType::ELU:
            return std::make_unique<ELU>(param > 0 ? param : 1.0);
        case ActivationType::SWISH:
            return std::make_unique<Swish>();
        case ActivationType::GELU:
            return std::make_unique<GELU>();
        case ActivationType::SELU:
            return std::make_unique<SELU>();
        case ActivationType::MISH:
            return std::make_unique<Mish>();
        case ActivationType::SOFTMAX:
            return std::make_unique<Softmax>();
        default:
            throw std::runtime_error("Unsupported activation function");
    }
}

// Sigmoid implementation
double Sigmoid::activate(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::derivative(double x) const {
    double s = activate(x);
    return s * (1.0 - s);
}

// Tanh implementation
double Tanh::derivative(double x) const {
    double t = std::tanh(x);
    return 1.0 - t * t;
}

// ELU implementation
double ELU::activate(double x) const {
    return x > 0.0 ? x : alpha * (std::exp(x) - 1.0);
}

double ELU::derivative(double x) const {
    return x > 0.0 ? 1.0 : alpha * std::exp(x);
}

// Swish implementation
double Swish::activate(double x) const {
    return x / (1.0 + std::exp(-x));
}

double Swish::derivative(double x) const {
    double sigmoid = 1.0 / (1.0 + std::exp(-x));
    return sigmoid + x * sigmoid * (1.0 - sigmoid);
}

// GELU implementation
double GELU::activate(double x) const {
    const double sqrt_2_pi = std::sqrt(2.0 / M_PI);
    double inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(inner));
}

double GELU::derivative(double x) const {
    const double sqrt_2_pi = std::sqrt(2.0 / M_PI);
    double x_cubed = x * x * x;
    double inner = sqrt_2_pi * (x + 0.044715 * x_cubed);
    double tanh_inner = std::tanh(inner);
    double sech2_inner = 1.0 - tanh_inner * tanh_inner;
    
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x);
}

// SELU implementation
double SELU::activate(double x) const {
    const double alpha = 1.6732632423543772848170429916717;
    const double scale = 1.0507009873554804934193349852946;
    return scale * (x > 0.0 ? x : alpha * (std::exp(x) - 1.0));
}

double SELU::derivative(double x) const {
    const double alpha = 1.6732632423543772848170429916717;
    const double scale = 1.0507009873554804934193349852946;
    return scale * (x > 0.0 ? 1.0 : alpha * std::exp(x));
}

// Mish implementation
double Mish::activate(double x) const {
    return x * std::tanh(std::log(1.0 + std::exp(x)));
}

double Mish::derivative(double x) const {
    double exp_x = std::exp(x);
    double exp_2x = exp_x * exp_x;
    double exp_3x = exp_2x * exp_x;
    double softplus = std::log(1.0 + exp_x);
    double tanh_softplus = std::tanh(softplus);
    
    double numerator = exp_x * (4.0 * (x + 1.0) + 4.0 * exp_2x + exp_3x + exp_x * (4.0 * x + 6.0));
    double denominator = 2.0 * exp_x + exp_2x + 2.0;
    denominator = denominator * denominator;
    
    return tanh_softplus + x * numerator / denominator;
}

// Softmax implementation
std::vector<double> Softmax::activate(const std::vector<double>& inputs) const {
    std::vector<double> result(inputs.size());
    double max_val = *std::max_element(inputs.begin(), inputs.end());
    double sum = 0.0;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        result[i] = std::exp(inputs[i] - max_val);
        sum += result[i];
    }
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        result[i] /= sum;
    }
    
    return result;
}

std::vector<double> Softmax::derivative(const std::vector<double>& inputs) const {
    std::vector<double> result(inputs.size(), 1.0);
    return result; // Simplified - actual jacobian would be more complex
}

// Static utility functions for backward compatibility
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
            return swish(x);
        case ActivationType::GELU:
            return gelu(x);
        case ActivationType::MISH:
            return mish(x);
        default:
            throw std::runtime_error("Unsupported activation function");
    }
}

std::vector<double> Activation::apply(const std::vector<double>& x, ActivationType type) {
    std::vector<double> result(x.size());
    
    if (type == ActivationType::SOFTMAX) {
        return softmax(x);
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
        case ActivationType::SWISH:
            return swishDerivative(x);
        case ActivationType::GELU:
            return geluDerivative(x);
        case ActivationType::MISH:
            return mishDerivative(x);
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

// Advanced activation function implementations
double Activation::leakyRelu(double x, double alpha) {
    return x > 0.0 ? x : alpha * x;
}

double Activation::leakyReluDerivative(double x, double alpha) {
    return x > 0.0 ? 1.0 : alpha;
}

double Activation::elu(double x, double alpha) {
    return x > 0.0 ? x : alpha * (std::exp(x) - 1.0);
}

double Activation::eluDerivative(double x, double alpha) {
    return x > 0.0 ? 1.0 : alpha * std::exp(x);
}

double Activation::swish(double x) {
    return x / (1.0 + std::exp(-x));
}

double Activation::swishDerivative(double x) {
    double sigmoid = 1.0 / (1.0 + std::exp(-x));
    return sigmoid + x * sigmoid * (1.0 - sigmoid);
}

double Activation::gelu(double x) {
    const double sqrt_2_pi = std::sqrt(2.0 / M_PI);
    double inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(inner));
}

double Activation::geluDerivative(double x) {
    const double sqrt_2_pi = std::sqrt(2.0 / M_PI);
    double x_cubed = x * x * x;
    double inner = sqrt_2_pi * (x + 0.044715 * x_cubed);
    double tanh_inner = std::tanh(inner);
    double sech2_inner = 1.0 - tanh_inner * tanh_inner;
    
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x);
}

double Activation::mish(double x) {
    return x * std::tanh(std::log(1.0 + std::exp(x)));
}

double Activation::mishDerivative(double x) {
    double exp_x = std::exp(x);
    double exp_2x = exp_x * exp_x;
    double exp_3x = exp_2x * exp_x;
    double softplus = std::log(1.0 + exp_x);
    double tanh_softplus = std::tanh(softplus);
    
    double numerator = exp_x * (4.0 * (x + 1.0) + 4.0 * exp_2x + exp_3x + exp_x * (4.0 * x + 6.0));
    double denominator = 2.0 * exp_x + exp_2x + 2.0;
    denominator = denominator * denominator;
    
    return tanh_softplus + x * numerator / denominator;
}

std::vector<double> Activation::softmax(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    double max_val = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }
    
    return result;
}

std::vector<double> Activation::softmaxDerivative(const std::vector<double>& x, const std::vector<double>& outputGradient) {
    std::vector<double> softmax_output = softmax(x);
    std::vector<double> result(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            if (i == j) {
                result[i] += outputGradient[j] * softmax_output[i] * (1.0 - softmax_output[i]);
            } else {
                result[i] -= outputGradient[j] * softmax_output[i] * softmax_output[j];
            }
        }
    }
    
    return result;
}
