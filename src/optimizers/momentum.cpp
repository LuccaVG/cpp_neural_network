#include "momentum.h"

Momentum::Momentum(double learningRate, double momentum)
    : learningRate(learningRate), momentum(momentum) {}

void Momentum::update(std::vector<double>& weights, 
                      const std::vector<double>& gradients,
                      int iteration) {
    // Initialize velocity vector if needed
    if (velocity.size() != weights.size()) {
        velocity.resize(weights.size(), 0.0);
    }
    
    // Update with momentum
    for (size_t i = 0; i < weights.size(); ++i) {
        velocity[i] = momentum * velocity[i] - learningRate * gradients[i];
        weights[i] += velocity[i];
    }
}

std::unique_ptr<Optimizer> Momentum::clone() const {
    auto clone = std::make_unique<Momentum>(learningRate, momentum);
    clone->velocity = velocity;
    return clone;
}

void Momentum::reset() {
    std::fill(velocity.begin(), velocity.end(), 0.0);
}

OptimizerType Momentum::getType() const {
    return OptimizerType::MOMENTUM;
}