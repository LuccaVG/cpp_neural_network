#include <iostream>
#include <vector>
#include "neural_network.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "initializer.h"

int main() {
    // Define the XOR input and output
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> outputs = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
    NeuralNetwork nn;
    nn.addLayer(LayerType::DENSE, 2, 2, ActivationType::TANH);
    nn.addLayer(LayerType::DENSE, 2, 1, ActivationType::SIGMOID);

    // Set the optimizer
    auto optimizer = Optimizer::create(OptimizerType::ADAM, 0.01);
    nn.setOptimizer(std::move(optimizer));

    // Train the neural network
    for (int epoch = 0; epoch < 10000; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            nn.train(inputs[i], outputs[i]);
        }
    }

    // Test the neural network
    std::cout << "XOR Predictions:" << std::endl;
    for (const auto& input : inputs) {
        auto prediction = nn.predict(input);
        std::cout << input[0] << ", " << input[1] << " => " << prediction[0] << std::endl;
    }

    return 0;
}