#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <iostream>
#include "core/types.h"
#include "layers/layer.h"
#include "optimizers/optimizer.h"
#include "core/loss.h"

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void addLayer(std::unique_ptr<Layer> layer);
    void compile(OptimizerType optimizerType, LossType lossType, double learningRate);
    void fit(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, int epochs, int batchSize);
    void train(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, int epochs, int batchSize) {
        fit(x, y, epochs, batchSize);
    }
    std::vector<double> predict(const std::vector<double>& input);
    void save(const std::string& filename) const;
    void load(const std::string& filename);

protected:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<Loss> loss;
    LossType lossType;

    void forward(const std::vector<double>& input);
    void backward(const std::vector<double>& outputGradient);
    void updateParameters(int iteration);
};

#endif // NEURAL_NETWORK_H