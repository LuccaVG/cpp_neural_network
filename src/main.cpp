#include <iostream>
#include "neural_network.h"

int main() {
    // Initialize the neural network
    NeuralNetwork nn;

    // Load data (this is a placeholder, implement your data loading logic)
    std::vector<std::vector<double>> trainingData; // Load your training data here
    std::vector<std::vector<double>> trainingLabels; // Load your training labels here

    // Train the neural network
    nn.train(trainingData, trainingLabels, 1000, 0.01); // 1000 epochs, learning rate of 0.01

    // Test the neural network (this is a placeholder, implement your testing logic)
    std::vector<std::vector<double>> testData; // Load your test data here
    auto predictions = nn.predict(testData);

    // Output predictions (this is a placeholder, implement your output logic)
    for (const auto& prediction : predictions) {
        for (const auto& value : prediction) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}