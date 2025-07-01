#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include "neural_network.h"
#include "layers/dense_layer.h"
#include "core/types.h"

// Generate XOR dataset for testing
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generateXORDataset() {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    
    // XOR truth table - repeat for more training samples
    for (int repeat = 0; repeat < 250; ++repeat) {
        inputs.push_back({0, 0});
        outputs.push_back({0});
        inputs.push_back({0, 1});
        outputs.push_back({1});
        inputs.push_back({1, 0});
        outputs.push_back({1});
        inputs.push_back({1, 1});
        outputs.push_back({0});
    }
    
    return {inputs, outputs};
}

void testOptimizer(OptimizerType optimizerType, const std::string& name) {
    std::cout << "\nTesting " << name << ":" << std::endl;
    
    try {
        // Create fresh network for each optimizer
        NeuralNetwork nn;
        nn.addLayer(std::make_unique<DenseLayer>(2, 6, ActivationType::TANH));
        nn.addLayer(std::make_unique<DenseLayer>(6, 1, ActivationType::SIGMOID));
        nn.compile(optimizerType, LossType::MEAN_SQUARED_ERROR, 0.01);
        
        auto [trainX, trainY] = generateXORDataset();
        
        std::cout << "  Training..." << std::flush;
        nn.fit(trainX, trainY, 2000, trainX.size());
        
        // Test the network
        std::vector<std::vector<double>> testInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        std::vector<std::vector<double>> expectedOutputs = {{0}, {1}, {1}, {0}};
        
        int correct = 0;
        std::cout << "\n  Results:" << std::endl;
        for (size_t i = 0; i < testInputs.size(); ++i) {
            auto prediction = nn.predict(testInputs[i]);
            bool predicted = prediction[0] > 0.5;
            bool expected = expectedOutputs[i][0] > 0.5;
            if (predicted == expected) correct++;
            
            std::cout << "    [" << testInputs[i][0] << ", " << testInputs[i][1] 
                      << "] -> " << std::fixed << std::setprecision(6) << prediction[0] 
                      << " (expected " << expectedOutputs[i][0] << ")" << std::endl;
        }
        
        double accuracy = static_cast<double>(correct) / testInputs.size();
        std::cout << "  Accuracy: " << std::fixed << std::setprecision(1) << accuracy * 100 << "%" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  COMPREHENSIVE OPTIMIZER TEST" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test all optimizers
    testOptimizer(OptimizerType::SGD, "SGD");
    testOptimizer(OptimizerType::ADAM, "Adam");
    testOptimizer(OptimizerType::MOMENTUM, "Momentum");
    testOptimizer(OptimizerType::RMSPROP, "RMSProp");
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "   ALL OPTIMIZER TESTS COMPLETED!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
