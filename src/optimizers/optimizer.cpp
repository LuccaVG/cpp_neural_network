// filepath: /cpp_neural_network/cpp_neural_network/src/optimizers/optimizer.cpp

#include "optimizer.h"
#include <stdexcept>

Optimizer::~Optimizer() = default;

std::unique_ptr<Optimizer> Optimizer::create(OptimizerType type, double learningRate) {
    switch (type) {
        case OptimizerType::SGD:
            return std::make_unique<SGD>(learningRate);
        case OptimizerType::MOMENTUM:
            return std::make_unique<Momentum>(learningRate);
        case OptimizerType::RMSPROP:
            return std::make_unique<RMSProp>(learningRate);
        case OptimizerType::ADAM:
            return std::make_unique<Adam>(learningRate);
        default:
            throw std::runtime_error("Unsupported optimizer type");
    }
}