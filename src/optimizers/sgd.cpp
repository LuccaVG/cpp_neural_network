// filepath: /cpp_neural_network/cpp_neural_network/src/optimizers/sgd.cpp
#include "sgd.h"

SGD::SGD(double learningRate) : learningRate(learningRate) {}

void SGD::update(std::vector<double>& weights, const std::vector<double>& gradients, int iteration) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learningRate * gradients[i];
    }
}

std::unique_ptr<Optimizer> SGD::clone() const {
    return std::make_unique<SGD>(learningRate);
}

void SGD::reset() {
    // No state to reset for SGD
}

OptimizerType SGD::getType() const {
    return OptimizerType::SGD;
}