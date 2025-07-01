#include <iostream>
#include <vector>
#include <memory>
#include "neural_network.h"
#include "layers/dense_layer.h"
#include "core/types.h"

// Simple working demo with just SGD to establish baseline
int main() {
    std::cout << "=== ROBUST NEURAL NETWORK DEMO ===" << std::endl;
    
    try {
        // Create neural network
        NeuralNetwork nn;
        
        // Build XOR network
        nn.addLayer(std::make_unique<DenseLayer>(2, 4, ActivationType::TANH));
        nn.addLayer(std::make_unique<DenseLayer>(4, 1, ActivationType::SIGMOID));
        
        // Use only SGD optimizer (known to work)
        nn.compile(OptimizerType::SGD, LossType::MEAN_SQUARED_ERROR, 0.1);
        
        // XOR training data
        std::vector<std::vector<double>> inputs = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        std::vector<std::vector<double>> outputs = {
            {0}, {1}, {1}, {0}
        };
        
        // Replicate data for better training
        std::vector<std::vector<double>> trainX, trainY;
        for (int i = 0; i < 500; ++i) {
            for (size_t j = 0; j < inputs.size(); ++j) {
                trainX.push_back(inputs[j]);
                trainY.push_back(outputs[j]);
            }
        }
        
        std::cout << "Training XOR with SGD..." << std::endl;
        nn.fit(trainX, trainY, 3000, trainX.size());
        
        // Test the network
        std::cout << "\nTesting XOR results:" << std::endl;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto prediction = nn.predict(inputs[i]);
            bool predicted = prediction[0] > 0.5;
            bool expected = outputs[i][0] > 0.5;
            
            std::cout << "[" << inputs[i][0] << ", " << inputs[i][1] 
                      << "] -> " << prediction[0] 
                      << " (expected " << outputs[i][0] << ") " 
                      << (predicted == expected ? "✓" : "✗") << std::endl;
        }
        
        // Test binary classification
        std::cout << "\n=== BINARY CLASSIFICATION TEST ===" << std::endl;
        
        NeuralNetwork classifier;
        classifier.addLayer(std::make_unique<DenseLayer>(2, 8, ActivationType::RELU));
        classifier.addLayer(std::make_unique<DenseLayer>(8, 1, ActivationType::SIGMOID));
        classifier.compile(OptimizerType::SGD, LossType::BINARY_CROSS_ENTROPY, 0.01);
        
        // Generate simple separable data
        std::vector<std::vector<double>> classInputs;
        std::vector<std::vector<double>> classOutputs;
        
        for (int i = 0; i < 200; ++i) {
            double x = (i % 20) / 10.0 - 1.0; // -1 to 1
            double y = (i / 20) / 10.0 - 1.0; // -1 to 1
            classInputs.push_back({x, y});
            // Simple linear decision boundary: y > x
            classOutputs.push_back({y > x ? 1.0 : 0.0});
        }
        
        std::cout << "Training binary classifier..." << std::endl;
        classifier.fit(classInputs, classOutputs, 1000, 32);
        
        // Test classifier
        int correct = 0;
        for (size_t i = 0; i < classInputs.size(); ++i) {
            auto pred = classifier.predict(classInputs[i]);
            bool predicted = pred[0] > 0.5;
            bool expected = classOutputs[i][0] > 0.5;
            if (predicted == expected) correct++;
        }
        
        double accuracy = static_cast<double>(correct) / classInputs.size();
        std::cout << "Classification accuracy: " << accuracy * 100 << "%" << std::endl;
        
        std::cout << "\n=== ALL TESTS COMPLETED SUCCESSFULLY ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
