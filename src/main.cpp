#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include "neural_network.h"
#include "layers/dense_layer.h"
#include "core/types.h"

// Helper function to print vectors
void printVector(const std::vector<double>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    // Create a neural network
    NeuralNetwork nn;

    // Add layers
    nn.addLayer(std::make_unique<DenseLayer>(2, 4, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(4, 1, ActivationType::SIGMOID));

    // Compile the network
    nn.compile(OptimizerType::ADAM, LossType::MEAN_SQUARED_ERROR, 0.01);

    // Create XOR training data
    std::vector<std::vector<double>> xorInputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    
    std::vector<std::vector<double>> xorOutputs = {
        {0},
        {1},
        {1},
        {0}
    };

    // Train the network
    std::cout << "Training the network..." << std::endl;
    nn.fit(xorInputs, xorOutputs, 10000, 4);

    // Test the network
    std::cout << "\nTesting the network:" << std::endl;
    for (size_t i = 0; i < xorInputs.size(); ++i) {
        // Fix: Using predict with a single sample (1D vector)
        std::vector<double> prediction = nn.predict(xorInputs[i]);
        
        std::cout << "Input: [" << xorInputs[i][0] << ", " << xorInputs[i][1] << "] ";
        std::cout << "Predicted: ";
        printVector(prediction);
        std::cout << "Expected: [" << xorOutputs[i][0] << "]" << std::endl;
    }

    // Save the trained model
    nn.save("xor_model.dat");
    std::cout << "\nModel saved to xor_model.dat" << std::endl;

    std::cout << "\nTrying to load and use the model:" << std::endl;
    
    // Create a new network and load the saved model
    NeuralNetwork loadedNN;
    loadedNN.load("xor_model.dat");
    
    // Test the loaded network
    for (size_t i = 0; i < xorInputs.size(); ++i) {
        std::vector<double> prediction = loadedNN.predict(xorInputs[i]);
        
        std::cout << "Input: [" << xorInputs[i][0] << ", " << xorInputs[i][1] << "] ";
        std::cout << "Predicted: ";
        printVector(prediction);
    }

    return 0;
}