#include "loss.h"
#include <cmath>
#include <stdexcept>
#include <vector>

double Loss::calculate(const std::vector<double>& predicted, 
                       const std::vector<double>& target,
                       LossType type) {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target size mismatch");
    }
    
    switch (type) {
        case LossType::MEAN_SQUARED_ERROR: {
            double sum = 0.0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                double diff = predicted[i] - target[i];
                sum += diff * diff;
            }
            return sum / predicted.size();
        }
        
        case LossType::BINARY_CROSS_ENTROPY: {
            double sum = 0.0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                double p = std::max(1e-10, std::min(1.0 - 1e-10, predicted[i]));
                sum += target[i] * std::log(p) + (1.0 - target[i]) * std::log(1.0 - p);
            }
            return -sum / predicted.size();
        }
        
        case LossType::CATEGORICAL_CROSS_ENTROPY: {
            double sum = 0.0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                double p = std::max(1e-10, predicted[i]);
                sum += target[i] * std::log(p);
            }
            return -sum;
        }
        
        case LossType::HUBER_LOSS: {
            const double delta = 1.0;
            double sum = 0.0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                double diff = std::abs(predicted[i] - target[i]);
                if (diff <= delta) {
                    sum += 0.5 * diff * diff;
                } else {
                    sum += delta * (diff - 0.5 * delta);
                }
            }
            return sum / predicted.size();
        }
        
        case LossType::MEAN_ABSOLUTE_ERROR: {
            double sum = 0.0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                sum += std::abs(predicted[i] - target[i]);
            }
            return sum / predicted.size();
        }
        
        default:
            throw std::runtime_error("Unsupported loss function");
    }
}

std::vector<double> Loss::gradient(const std::vector<double>& predicted,
                                    const std::vector<double>& target,
                                    LossType type) {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target size mismatch");
    }
    
    std::vector<double> grad(predicted.size());
    
    switch (type) {
        case LossType::MEAN_SQUARED_ERROR:
            for (size_t i = 0; i < predicted.size(); ++i) {
                grad[i] = 2.0 * (predicted[i] - target[i]) / predicted.size();
            }
            break;
            
        case LossType::BINARY_CROSS_ENTROPY:
            for (size_t i = 0; i < predicted.size(); ++i) {
                double p = std::max(1e-10, std::min(1.0 - 1e-10, predicted[i]));
                grad[i] = (p - target[i]) / (p * (1.0 - p) * predicted.size());
            }
            break;
            
        case LossType::CATEGORICAL_CROSS_ENTROPY:
            for (size_t i = 0; i < predicted.size(); ++i) {
                grad[i] = (predicted[i] - target[i]);
            }
            break;
            
        case LossType::HUBER_LOSS: {
            const double delta = 1.0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                double diff = predicted[i] - target[i];
                if (std::abs(diff) <= delta) {
                    grad[i] = diff / predicted.size();
                } else {
                    grad[i] = ((diff > 0.0) ? delta : -delta) / predicted.size();
                }
            }
            break;
        }
            
        case LossType::MEAN_ABSOLUTE_ERROR:
            for (size_t i = 0; i < predicted.size(); ++i) {
                double diff = predicted[i] - target[i];
                grad[i] = ((diff > 0.0) ? 1.0 : -1.0) / predicted.size();
            }
            break;
            
        default:
            throw std::runtime_error("Unsupported loss function gradient");
    }
    
    return grad;
}