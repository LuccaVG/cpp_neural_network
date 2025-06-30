#include "loss.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

double Loss::calculate(const std::vector<double>& predicted, const std::vector<double>& target) const {
    // This should be a pure virtual function, so implementation should be in derived classes
    throw std::runtime_error("calculate() called on base Loss class");
}

std::vector<double> Loss::calculateGradient(const std::vector<double>& predicted, const std::vector<double>& target) const {
    // This should be a pure virtual function, so implementation should be in derived classes
    throw std::runtime_error("calculateGradient() called on base Loss class");
}

// MeanSquaredError implementation
double MeanSquaredError::calculate(const std::vector<double>& predicted, const std::vector<double>& target) const {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target sizes don't match");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        double diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

std::vector<double> MeanSquaredError::calculateGradient(const std::vector<double>& predicted, const std::vector<double>& target) const {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target sizes don't match");
    }
    
    std::vector<double> grad(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        grad[i] = 2.0 * (predicted[i] - target[i]) / predicted.size();
    }
    return grad;
}

std::vector<double> MeanSquaredError::gradient(const std::vector<double>& predicted, const std::vector<double>& target) const {
    return calculateGradient(predicted, target);
}

// BinaryCrossEntropy implementation
double BinaryCrossEntropy::calculate(const std::vector<double>& predicted, const std::vector<double>& target) const {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target sizes don't match");
    }
    
    const double epsilon = 1e-10; // To avoid log(0)
    double sum = 0.0;
    
    for (size_t i = 0; i < predicted.size(); ++i) {
        double p = std::max(std::min(predicted[i], 1.0 - epsilon), epsilon);
        sum += target[i] * std::log(p) + (1.0 - target[i]) * std::log(1.0 - p);
    }
    
    return -sum / predicted.size();
}

std::vector<double> BinaryCrossEntropy::calculateGradient(const std::vector<double>& predicted, const std::vector<double>& target) const {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target sizes don't match");
    }
    
    const double epsilon = 1e-10;
    std::vector<double> grad(predicted.size());
    
    for (size_t i = 0; i < predicted.size(); ++i) {
        double p = std::max(std::min(predicted[i], 1.0 - epsilon), epsilon);
        grad[i] = -(target[i] / p - (1.0 - target[i]) / (1.0 - p)) / predicted.size();
    }
    
    return grad;
}

std::vector<double> BinaryCrossEntropy::gradient(const std::vector<double>& predicted, const std::vector<double>& target) const {
    return calculateGradient(predicted, target);
}

// Factory method implementation
std::unique_ptr<Loss> Loss::create(LossType type) {
    switch (type) {
        case LossType::MEAN_SQUARED_ERROR:
            return std::make_unique<MeanSquaredError>();
        case LossType::BINARY_CROSS_ENTROPY:
            return std::make_unique<BinaryCrossEntropy>();
        default:
            throw std::runtime_error("Unsupported loss type");
    }
}