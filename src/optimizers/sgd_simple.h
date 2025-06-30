#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

class SGD : public Optimizer {
private:
    double learningRate;

public:
    SGD(double learningRate = 0.01) : learningRate(learningRate) {}

    void update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * gradients[i];
        }
    }
};

#endif // SGD_H
