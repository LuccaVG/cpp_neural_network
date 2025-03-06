// filepath: /cpp_neural_network/cpp_neural_network/src/layers/dropout_layer.h

#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include <vector>
#include <random>

class DropoutLayer : public Layer {
private:
    double dropoutRate;
    std::vector<double> mask;
    std::mt19937 rng;

public:
    DropoutLayer(double rate);
    
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

#endif // DROPOUT_LAYER_H