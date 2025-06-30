#ifndef ADAM_SIMPLE_H
#define ADAM_SIMPLE_H

#include "optimizer.h"

class Adam : public Optimizer {
private:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;

public:
    Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        // Simplified Adam implementation - just use SGD for now
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * gradients[i];
        }
    }
};

#endif // ADAM_SIMPLE_H
