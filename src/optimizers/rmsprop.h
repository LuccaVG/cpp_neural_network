// filepath: /cpp_neural_network/cpp_neural_network/src/optimizers/rmsprop.h

#ifndef RMSPROP_H
#define RMSPROP_H

#include "optimizer.h"
#include <vector>

class RMSProp : public Optimizer {
private:
    double learningRate;
    double decay;
    double epsilon;
    std::vector<double> cache;

public:
    RMSProp(double learningRate = 0.001, double decay = 0.9, double epsilon = 1e-8);

    void update(std::vector<double>& weights, 
                const std::vector<double>& gradients,
                int iteration) override;

    std::unique_ptr<Optimizer> clone() const override;

    void reset() override;

    OptimizerType getType() const override;
};

#endif // RMSPROP_H