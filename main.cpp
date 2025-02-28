// Remove #pragma once - this is only for header files

#include <iostream>
#include <vector>
#include "neural_network.h"

// Define a proper main function
int main() {
    std::cout << "Neural Network Prototype" << std::endl;
    
    // Example usage:
    // Define network architecture (input layer -> hidden layer -> output layer)
    std::vector<int> layers = {2, 4, 1};  // 2 inputs, 4 neurons in hidden layer, 1 output
    
    // Create neural network with ReLU for hidden layers and sigmoid for output
    NeuralNetwork network(
        layers, 
        NeuralNetwork::ActivationFunction::RELU, 
        NeuralNetwork::ActivationFunction::SIGMOID
    );
    
    // Define some training data (XOR problem)
    std::vector<std::vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    
    std::vector<std::vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };
    
    // Set up training options
    NeuralNetwork::TrainingOptions options;
    options.epochs = 1000;
    options.learningRate = 0.01;
    options.optimizer = NeuralNetwork::Optimizer::ADAM;
    options.batchSize = 4;  // Full batch
    
    // Train network
    std::cout << "Training neural network..." << std::endl;
    network.train(inputs, targets, options);
    
    // Evaluate network
    double accuracy = network.evaluateAccuracy(inputs, targets);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    // Test network
    std::cout << "\nTesting network predictions:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> output = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "]"
                  << " -> Expected: " << targets[i][0]
                  << ", Predicted: " << output[0] << std::endl;
    }
    
    // Print model summary
    std::cout << "\n" << network.getModelSummary() << std::endl;
    
    return 0;
}