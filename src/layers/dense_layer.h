// filepath: /cpp_neural_network/cpp_neural_network/src/layers/dense_layer.h

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "../core/activation.h"
#include "../core/initializer.h"
#include <vector>
#include <random>

class DenseLayer : public Layer {
private:
    size_t inputSize;
    size_t outputSize;
    ActivationType activation;
    InitializerType initializer;

    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> lastInput;
    std::vector<double> lastOutput;

public:
    DenseLayer(size_t inputSize, size_t outputSize, ActivationType activation = ActivationType::SIGMOID, InitializerType initializer = InitializerType::GLOROT_UNIFORM);

    std::vector<double> forward(const std::vector<double>& input, bool training) override;
    std::vector<double> backward(const std::vector<double>& outputGradient) override;
    void updateParameters(Optimizer& optimizer, int iteration) override;
    size_t getParameterCount() const override;
    size_t getOutputSize() const override;
    LayerType getType() const override;
    std::string getName() const override;
    void saveParameters(std::ofstream& file) const override;
    void loadParameters(std::ifstream& file) override;
    void reset() override;
};

#endif // DENSE_LAYER_H