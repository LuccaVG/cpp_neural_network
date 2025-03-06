// filepath: /cpp_neural_network/cpp_neural_network/src/optimizers/momentum.h

#ifndef MOMENTUM_H
#define MOMENTUM_H

#include "optimizer.h"
#include <vector>

class Momentum : public Optimizer {
private:
    double learningRate;
    double momentum;
    std::vector<double> velocity;

public:
    Momentum(double learningRate, double momentum = 0.9);

    void update(std::vector<double>& weights, 
                const std::vector<double>& gradients,
                int iteration) override;

    std::unique_ptr<Optimizer> clone() const override;

    void reset() override;

    OptimizerType getType() const override;
};

#endif // MOMENTUM_H