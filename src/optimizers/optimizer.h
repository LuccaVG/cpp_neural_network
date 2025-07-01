#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>
#include "../core/types.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) = 0;
    
    // Factory method to create optimizers based on type
    static std::unique_ptr<Optimizer> create(OptimizerType type, double learningRate = 0.01);
    
protected:
    // Utility function to clip gradients to prevent exploding gradients
    std::vector<double> clipGradients(const std::vector<double>& gradients, double maxNorm = 5.0) const {
        double norm = 0.0;
        for (double grad : gradients) {
            norm += grad * grad;
        }
        norm = std::sqrt(norm);
        
        if (norm > maxNorm) {
            std::vector<double> clipped(gradients.size());
            double scale = maxNorm / norm;
            for (size_t i = 0; i < gradients.size(); ++i) {
                clipped[i] = gradients[i] * scale;
            }
            return clipped;
        }
        return gradients;
    }
    
    // Check for NaN or infinite values
    bool isValidGradient(const std::vector<double>& gradients) const {
        for (double grad : gradients) {
            if (std::isnan(grad) || std::isinf(grad)) {
                return false;
            }
        }
        return true;
    }
};

// SGD implementation with gradient clipping
class SGD : public Optimizer {
public:
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        if (!isValidGradient(gradients)) {
            return; // Skip update if gradients are invalid
        }
        
        auto clippedGrads = clipGradients(gradients);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * clippedGrads[i];
        }
    }
};

// Robust Adam implementation with numerical stability improvements
class Adam : public Optimizer {
private:
    std::vector<double> m; // First moment (momentum)
    std::vector<double> v; // Second moment (RMSProp)
    double beta1;          // First moment decay rate
    double beta2;          // Second moment decay rate
    double epsilon;        // Small constant for numerical stability
    int t;                 // Time step for bias correction
    
public:
    Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}
    
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        if (!isValidGradient(gradients)) {
            return; // Skip update if gradients are invalid
        }
        
        auto clippedGrads = clipGradients(gradients);
        
        // Initialize moment vectors on first call
        if (m.empty()) {
            m.resize(weights.size(), 0.0);
            v.resize(weights.size(), 0.0);
        }
        
        t++; // Increment time step
        
        // Prevent overflow in bias correction
        double beta1_power = std::pow(beta1, t);
        double beta2_power = std::pow(beta2, t);
        
        // Add bounds to prevent numerical issues
        if (beta1_power < 1e-10) beta1_power = 1e-10;
        if (beta2_power < 1e-10) beta2_power = 1e-10;
        
        for (size_t i = 0; i < weights.size(); ++i) {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * clippedGrads[i];
            
            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * clippedGrads[i] * clippedGrads[i];
            
            // Compute bias-corrected first moment estimate
            double m_hat = m[i] / (1.0 - beta1_power);
            
            // Compute bias-corrected second raw moment estimate
            double v_hat = v[i] / (1.0 - beta2_power);
            
            // Add numerical stability checks
            double denominator = std::sqrt(v_hat) + epsilon;
            if (denominator < epsilon) denominator = epsilon;
            
            double update = learningRate * m_hat / denominator;
            
            // Limit the size of individual updates
            update = std::max(-1.0, std::min(1.0, update));
            
            // Update weights
            weights[i] -= update;
        }
    }
};

// Robust Momentum implementation
class Momentum : public Optimizer {
private:
    std::vector<double> velocity;
    double momentum;
    
public:
    Momentum(double momentum = 0.9) : momentum(momentum) {}
    
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        if (!isValidGradient(gradients)) {
            return; // Skip update if gradients are invalid
        }
        
        auto clippedGrads = clipGradients(gradients);
        
        // Initialize velocity on first call
        if (velocity.empty()) {
            velocity.resize(weights.size(), 0.0);
        }
        
        for (size_t i = 0; i < weights.size(); ++i) {
            // Update velocity: v = momentum * v - learningRate * gradient
            velocity[i] = momentum * velocity[i] - learningRate * clippedGrads[i];
            
            // Update weights: w = w + v
            weights[i] += velocity[i];
        }
    }
};

// Robust RMSProp implementation
class RMSProp : public Optimizer {
private:
    std::vector<double> squaredGradients;
    double decay;
    double epsilon;
    
public:
    RMSProp(double decay = 0.9, double epsilon = 1e-8) : decay(decay), epsilon(epsilon) {}
    
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        if (!isValidGradient(gradients)) {
            return; // Skip update if gradients are invalid
        }
        
        auto clippedGrads = clipGradients(gradients);
        
        // Initialize squared gradients on first call
        if (squaredGradients.empty()) {
            squaredGradients.resize(weights.size(), 0.0);
        }
        
        for (size_t i = 0; i < weights.size(); ++i) {
            // Accumulate squared gradients
            squaredGradients[i] = decay * squaredGradients[i] + (1.0 - decay) * clippedGrads[i] * clippedGrads[i];
            
            // Calculate adaptive learning rate with numerical stability
            double denominator = std::sqrt(squaredGradients[i]) + epsilon;
            if (denominator < epsilon) denominator = epsilon;
            
            double update = learningRate * clippedGrads[i] / denominator;
            
            // Limit the size of individual updates
            update = std::max(-1.0, std::min(1.0, update));
            
            // Update weights
            weights[i] -= update;
        }
    }
};

#endif // OPTIMIZER_H
