#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <memory>
#include "../core/types.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) = 0;
    
    // Factory method to create optimizers based on type
    static std::unique_ptr<Optimizer> create(OptimizerType type, double learningRate = 0.01);
};

// Simple SGD implementation
class SGD : public Optimizer {
public:
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * gradients[i];
        }
    }
};

// Simple Adam implementation (simplified to SGD for now)
class Adam : public Optimizer {
public:
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * gradients[i];
        }
    }
};

// Placeholder implementations for other optimizers
class Momentum : public Optimizer {
public:
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * gradients[i];
        }
    }
};

class RMSProp : public Optimizer {
public:
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * gradients[i];
        }
    }
};

#endif // OPTIMIZER_H
