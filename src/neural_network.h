// filepath: /cpp_neural_network/cpp_neural_network/src/neural_network.h

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include "core/types.h"
#include "layers/layer.h"
#include "optimizers/optimizer.h"

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void addLayer(std::unique_ptr<Layer> layer);
    void compile(OptimizerType optimizerType, double learningRate);
    void fit(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, int epochs, int batchSize);
    std::vector<double> predict(const std::vector<double>& input);
    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimizer> optimizer;
    LossType lossType;

    void forward(const std::vector<double>& input);
    void backward(const std::vector<double>& outputGradient);
    void updateParameters(int iteration);
};

#endif // NEURAL_NETWORK_H