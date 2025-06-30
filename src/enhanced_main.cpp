#include <iostream>
#include <vector>
#include <memory>
#include "neural_network.h"
#include "layers/dense_layer.h"
#include "core/types.h"

int main() {
    std::cout << "Enhanced Neural Network Demo" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Create neural network (using regular NeuralNetwork for now)
    NeuralNetwork nn;
    
    // Build a network
    nn.addLayer(std::make_unique<DenseLayer>(2, 8, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(8, 4, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(4, 1, ActivationType::SIGMOID));
    
    // Compile the network
    nn.compile(OptimizerType::ADAM, LossType::MEAN_SQUARED_ERROR, 0.001);
    
    // Create XOR training data
    std::vector<std::vector<double>> xorInputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
        {0.1, 0.1}, {0.9, 0.1}, {0.1, 0.9}, {0.9, 0.9}  // Add some noise
    };
    
    std::vector<std::vector<double>> xorOutputs = {
        {0}, {1}, {1}, {0},
        {0}, {1}, {1}, {0}
    };
    
    // Train the network
    std::cout << "\nTraining enhanced network..." << std::endl;
    nn.fit(xorInputs, xorOutputs, 5000, 8);
    
    // Test individual predictions
    std::cout << "\nTesting XOR predictions:" << std::endl;
    std::vector<std::vector<double>> testInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> expectedOutputs = {{0}, {1}, {1}, {0}};
    
    for (size_t i = 0; i < testInputs.size(); ++i) {
        auto prediction = nn.predict(testInputs[i]);
        std::cout << "Input: [" << testInputs[i][0] << ", " << testInputs[i][1] 
                  << "] -> Predicted: " << prediction[0] 
                  << " (Expected: " << expectedOutputs[i][0] << ")" << std::endl;
    }
    
    // Calculate accuracy manually
    int correct = 0;
    for (size_t i = 0; i < testInputs.size(); ++i) {
        auto prediction = nn.predict(testInputs[i]);
        bool predicted_class = prediction[0] > 0.5;
        bool actual_class = expectedOutputs[i][0] > 0.5;
        if (predicted_class == actual_class) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / testInputs.size();
    std::cout << "\nAccuracy: " << accuracy * 100 << "%" << std::endl;
    
    // Save the model
    nn.save("enhanced_xor_model.dat");
    std::cout << "Model saved to enhanced_xor_model.dat" << std::endl;
    
    return 0;
}
