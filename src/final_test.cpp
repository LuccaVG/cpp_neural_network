#include <iostream>
#include <vector>
#include <memory>
#include "neural_network.h"
#include "layers/dense_layer.h"
#include "core/types.h"

int main() {
    std::cout << "=== NEURAL NETWORK WITH ALL OPTIMIZERS TEST ===" << std::endl;
    
    // XOR training data
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> outputs = {
        {0}, {1}, {1}, {0}
    };
    
    // Replicate data for better training
    std::vector<std::vector<double>> trainX, trainY;
    for (int i = 0; i < 100; ++i) {
        for (size_t j = 0; j < inputs.size(); ++j) {
            trainX.push_back(inputs[j]);
            trainY.push_back(outputs[j]);
        }
    }
    
    // Test all optimizers
    std::vector<OptimizerType> optimizers = {
        OptimizerType::SGD, 
        OptimizerType::ADAM, 
        OptimizerType::MOMENTUM, 
        OptimizerType::RMSPROP
    };
    
    std::vector<std::string> optimizerNames = {
        "SGD", "Adam", "Momentum", "RMSProp"
    };
    
    for (size_t i = 0; i < optimizers.size(); ++i) {
        std::cout << "\nTesting " << optimizerNames[i] << " optimizer:" << std::endl;
        
        try {
            NeuralNetwork nn;
            nn.addLayer(std::make_unique<DenseLayer>(2, 4, ActivationType::TANH));
            nn.addLayer(std::make_unique<DenseLayer>(4, 1, ActivationType::SIGMOID));
            nn.compile(optimizers[i], LossType::MEAN_SQUARED_ERROR, 0.01);
            
            // Quick training
            nn.fit(trainX, trainY, 1000, trainX.size());
            
            // Test results
            int correct = 0;
            for (size_t j = 0; j < inputs.size(); ++j) {
                auto result = nn.predict(inputs[j]);
                bool isCorrect = (result[0] > 0.5) == (outputs[j][0] > 0.5);
                if (isCorrect) correct++;
                std::cout << "[" << inputs[j][0] << ", " << inputs[j][1] << "] -> " 
                         << result[0] << " (expected " << outputs[j][0] << ") " 
                         << (isCorrect ? "✓" : "✗") << std::endl;
            }
            
            double accuracy = static_cast<double>(correct) / inputs.size();
            std::cout << optimizerNames[i] << " Accuracy: " << accuracy * 100 << "%" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error with " << optimizerNames[i] << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << "\n=== ALL OPTIMIZER TESTS COMPLETED ===" << std::endl;
    
    return 0;
}
