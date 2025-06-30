#ifndef ADVANCED_OPTIMIZER_H
#define ADVANCED_OPTIMIZER_H

#include "optimizer.h"
#include <cmath>
#include <algorithm>

// Learning rate scheduler base class
class LearningRateScheduler {
public:
    virtual ~LearningRateScheduler() = default;
    virtual double getLearningRate(int epoch, double baseLearningRate) = 0;
};

// Exponential decay scheduler
class ExponentialDecay : public LearningRateScheduler {
private:
    double decayRate;
    int decaySteps;
    
public:
    ExponentialDecay(double decayRate = 0.96, int decaySteps = 1000) 
        : decayRate(decayRate), decaySteps(decaySteps) {}
    
    double getLearningRate(int epoch, double baseLearningRate) override {
        return baseLearningRate * std::pow(decayRate, epoch / decaySteps);
    }
};

// Step decay scheduler
class StepDecay : public LearningRateScheduler {
private:
    double factor;
    int stepSize;
    
public:
    StepDecay(double factor = 0.1, int stepSize = 10) 
        : factor(factor), stepSize(stepSize) {}
    
    double getLearningRate(int epoch, double baseLearningRate) override {
        return baseLearningRate * std::pow(factor, epoch / stepSize);
    }
};

// Advanced Adam optimizer with proper momentum and adaptive learning rates
class AdvancedAdam : public Optimizer {
private:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    std::vector<double> m; // First moment
    std::vector<double> v; // Second moment
    int t; // Time step
    std::unique_ptr<LearningRateScheduler> scheduler;
    
public:
    AdvancedAdam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, 
                 double epsilon = 1e-8, std::unique_ptr<LearningRateScheduler> scheduler = nullptr)
        : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), 
          t(0), scheduler(std::move(scheduler)) {}
    
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double currentLearningRate) override {
        if (m.size() != weights.size()) {
            m.resize(weights.size(), 0.0);
            v.resize(weights.size(), 0.0);
        }
        
        t++; // Increment time step
        
        // Use scheduler if available
        double effectiveLR = scheduler ? scheduler->getLearningRate(t, learningRate) : currentLearningRate;
        
        for (size_t i = 0; i < weights.size(); ++i) {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * gradients[i];
            
            // Update biased second moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * gradients[i] * gradients[i];
            
            // Compute bias-corrected first moment estimate
            double m_hat = m[i] / (1.0 - std::pow(beta1, t));
            
            // Compute bias-corrected second moment estimate
            double v_hat = v[i] / (1.0 - std::pow(beta2, t));
            
            // Update weights
            weights[i] -= effectiveLR * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
    
    void setScheduler(std::unique_ptr<LearningRateScheduler> newScheduler) {
        scheduler = std::move(newScheduler);
    }
};

#endif // ADVANCED_OPTIMIZER_H
