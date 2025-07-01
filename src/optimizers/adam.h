// filepath: /cpp_neural_network/cpp_neural_network/src/optimizers/adam.h

#ifndef ADAM_H
#define ADAM_H

#include "optimizer.h"
#include <vector>

class Adam : public Optimizer {
private:
    double beta1;
    double beta2;
    double epsilon;
    std::vector<double> m;  // First moment estimate
    std::vector<double> v;  // Second moment estimate
    int t;  // Timestep

public:
    Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void update(std::vector<double>& weights, 
                const std::vector<double>& gradients,
                double learningRate) override;
};

#endif // ADAM_H