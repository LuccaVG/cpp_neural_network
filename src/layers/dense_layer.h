#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "../core/activation.h"
#include <vector>
#include <random>

class DenseLayer : public Layer {
private:
    size_t inputSize;
    size_t outputSize;
    ActivationType activation;
    
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> lastInput;
    std::vector<double> lastOutput;
    std::vector<double> lastActivatedOutput;
    std::vector<double> inputGradient;
    std::vector<double> lastDelta;

public:
    DenseLayer(size_t inputSize, size_t outputSize, ActivationType activation = ActivationType::SIGMOID);

    void forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient) override;
    void updateParameters(Optimizer& optimizer, int iteration) override;
    
    std::vector<double> getOutput() const override;
    std::vector<double> getInputGradient() const override;
    void save(std::ofstream& file) const override;
    void load(std::ifstream& file) override;
};

#endif // DENSE_LAYER_H
