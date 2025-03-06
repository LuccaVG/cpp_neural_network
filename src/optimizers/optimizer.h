#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <memory>
#include <cmath>
#include "../core/types.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;

    // Factory method to create optimizers based on type
    static std::unique_ptr<Optimizer> create(OptimizerType type, double learningRate);

    virtual void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) = 0;
};

class SGD : public Optimizer {
public:
    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * gradients[i];
        }
    }
};

class Momentum : public Optimizer {
private:
    std::vector<double> velocity;
    double beta;

public:
    Momentum(double beta = 0.9) : beta(beta) {}

    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        if (velocity.size() != weights.size()) {
            velocity.resize(weights.size(), 0.0);
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            velocity[i] = beta * velocity[i] + (1.0 - beta) * gradients[i];
            weights[i] -= learningRate * velocity[i];
        }
    }
};

class Adam : public Optimizer {
private:
    std::vector<double> m;
    std::vector<double> v;
    double beta1;
    double beta2;
    double epsilon;
    int t;

public:
    Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) 
        : beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        if (m.size() != weights.size()) {
            m.resize(weights.size(), 0.0);
            v.resize(weights.size(), 0.0);
        }

        t++;
        
        for (size_t i = 0; i < weights.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * gradients[i] * gradients[i];
            
            double m_hat = m[i] / (1.0 - std::pow(beta1, t));
            double v_hat = v[i] / (1.0 - std::pow(beta2, t));
            
            weights[i] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
};

// Implementation of the factory method
std::unique_ptr<Optimizer> Optimizer::create(OptimizerType type, double learningRate) {
    switch (type) {
        case OptimizerType::SGD:
            return std::make_unique<SGD>();
        case OptimizerType::MOMENTUM:
            return std::make_unique<Momentum>();
        case OptimizerType::ADAM:
            return std::make_unique<Adam>();
        // Add other optimizer types as needed
        default:
            return std::make_unique<SGD>();
    }
}

#endif // OPTIMIZER_H