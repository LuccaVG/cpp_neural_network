#include <iostream>
#include <vector>
#include <memory>
#include "neural_network.h"
#include "layers/dense_layer.h"
#include "optimizers/adam.h"
#include "core/loss.h"

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    void compile(OptimizerType optimizerType, double learningRate, LossType lossType) {
        optimizer = Optimizer::create(optimizerType, learningRate);
        lossFunction = lossType;
    }

    void fit(const std::vector<std::vector<double>>& xTrain, const std::vector<std::vector<double>>& yTrain, int epochs, int batchSize) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < xTrain.size(); i += batchSize) {
                // Get batch
                std::vector<double> xBatch(xTrain.begin() + i, xTrain.begin() + std::min(i + batchSize, xTrain.size()));
                std::vector<double> yBatch(yTrain.begin() + i, yTrain.begin() + std::min(i + batchSize, yTrain.size()));

                // Forward pass
                std::vector<double> output = forward(xBatch);

                // Compute loss and gradients
                std::vector<double> lossGradient = Loss::gradient(output, yBatch, lossFunction);
                backward(lossGradient);
                updateParameters();
            }
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed." << std::endl;
        }
    }

    std::vector<double> predict(const std::vector<double>& input) {
        return forward(input);
    }

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimizer> optimizer;
    LossType lossFunction;

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> output = input;
        for (const auto& layer : layers) {
            output = layer->forward(output, false);
        }
        return output;
    }

    void backward(const std::vector<double>& lossGradient) {
        std::vector<double> gradient = lossGradient;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            gradient = (*it)->backward(gradient);
        }
    }

    void updateParameters() {
        for (const auto& layer : layers) {
            layer->updateParameters(*optimizer, 0);
        }
    }
};