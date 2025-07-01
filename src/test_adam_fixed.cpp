#include <iostream>
#include <vector>
#include <memory>
#include "neural_network.h"
#include "layers/dense_layer.h"
#include "core/types.h"

int main() {
    std::cout << "=== TESTING ADAM OPTIMIZER ===" << std::endl;
    
    try {
        // Create neural network with Adam
        NeuralNetwork nn;
        nn.addLayer(std::make_unique<DenseLayer>(2, 4, ActivationType::TANH));
        nn.addLayer(std::make_unique<DenseLayer>(4, 1, ActivationType::SIGMOID));
        nn.compile(OptimizerType::ADAM, LossType::MEAN_SQUARED_ERROR, 0.01);
        
        // XOR training data
        std::vector<std::vector<double>> inputs = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        std::vector<std::vector<double>> outputs = {
            {0}, {1}, {1}, {0}
        };
        
        // Replicate data for better training
        std::vector<std::vector<double>> trainX, trainY;
        for (int i = 0; i < 250; ++i) {
            for (size_t j = 0; j < inputs.size(); ++j) {
                trainX.push_back(inputs[j]);
                trainY.push_back(outputs[j]);
            }
        }
        
        std::cout << "Training XOR with Adam..." << std::endl;
        nn.fit(trainX, trainY, 2000, trainX.size());
        
        // Test the network
        std::cout << "\nTesting XOR results with Adam:" << std::endl;
        int correct = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto prediction = nn.predict(inputs[i]);
            bool predicted = prediction[0] > 0.5;
            bool expected = outputs[i][0] > 0.5;
            if (predicted == expected) correct++;
            
            std::cout << "[" << inputs[i][0] << ", " << inputs[i][1] 
                      << "] -> " << prediction[0] 
                      << " (expected " << outputs[i][0] << ") " 
                      << (predicted == expected ? "✓" : "✗") << std::endl;
        }
        
        double accuracy = static_cast<double>(correct) / inputs.size();
        std::cout << "Adam Accuracy: " << accuracy * 100 << "%" << std::endl;
        
        std::cout << "\n=== ADAM TEST COMPLETED SUCCESSFULLY ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
