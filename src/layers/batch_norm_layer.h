// filepath: /cpp_neural_network/cpp_neural_network/src/layers/batch_norm_layer.h

#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include "layer.h"
#include <vector>
#include <random>

class BatchNormLayer : public Layer {
private:
    size_t inputSize;
    double epsilon;
    std::vector<double> gamma; // Scale parameter
    std::vector<double> beta;  // Shift parameter
    std::vector<double> runningMean; // Running mean for inference
    std::vector<double> runningVar;  // Running variance for inference
    std::vector<double> lastInput; // Store last input for backward pass
    std::mt19937 rng;

public:
    BatchNormLayer(size_t inputSize, double epsilon = 1e-5);

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

#endif // BATCH_NORM_LAYER_H