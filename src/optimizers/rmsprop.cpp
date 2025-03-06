#include "rmsprop.h"
#include <cmath>
#include <algorithm>

RMSProp::RMSProp(double learningRate, double decay, double epsilon)
    : learningRate(learningRate), decay(decay), epsilon(epsilon) {}

void RMSProp::update(std::vector<double>& weights, const std::vector<double>& gradients, int iteration) {
    if (cache.size() != weights.size()) {
        cache.resize(weights.size(), 0.0);
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        cache[i] = decay * cache[i] + (1 - decay) * gradients[i] * gradients[i];
        weights[i] -= learningRate * gradients[i] / (std::sqrt(cache[i]) + epsilon);
    }
}

std::unique_ptr<Optimizer> RMSProp::clone() const {
    return std::make_unique<RMSProp>(learningRate, decay, epsilon);
}

void RMSProp::reset() {
    std::fill(cache.begin(), cache.end(), 0.0);
}

OptimizerType RMSProp::getType() const {
    return OptimizerType::RMSPROP;
}