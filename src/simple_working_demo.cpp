#include <iostream>
#include <vector>
#include <memory>
#include <random>
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

// Generate synthetic binary classification dataset
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generateClassificationDataset(int numSamples = 1000) {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < numSamples; ++i) {
        std::vector<double> features(4);
        for (int j = 0; j < 4; ++j) {
            features[j] = dist(gen);
        }
        
        // Create a non-linear classification boundary
        double decision = features[0] * features[1] + 0.5 * features[2] - 0.3 * features[3] + 0.2;
        std::vector<double> label = {decision > 0 ? 1.0 : 0.0};
        
        inputs.push_back(features);
        outputs.push_back(label);
    }
    
    return {inputs, outputs};
}

void demonstrateXORWithOptimizers() {
    std::cout << "\n========== XOR WITH DIFFERENT OPTIMIZERS ==========\n" << std::endl;
    
    auto [trainX, trainY] = generateXORDataset();
    
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
        std::cout << "\nTesting " << optimizerNames[i] << ":" << std::endl;
        
        NeuralNetwork nn;
        nn.addLayer(std::make_unique<DenseLayer>(2, 8, ActivationType::RELU));
        nn.addLayer(std::make_unique<DenseLayer>(8, 4, ActivationType::RELU));
        nn.addLayer(std::make_unique<DenseLayer>(4, 1, ActivationType::SIGMOID));
        
        nn.compile(optimizers[i], LossType::MEAN_SQUARED_ERROR, 0.01);
        
        std::cout << "  Training..." << std::flush;
        nn.fit(trainX, trainY, 2000, trainX.size());
        
        // Test the network
        std::vector<std::vector<double>> testInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        std::vector<std::vector<double>> expectedOutputs = {{0}, {1}, {1}, {0}};
        
        int correct = 0;
        std::cout << "\n  Results:" << std::endl;
        for (size_t j = 0; j < testInputs.size(); ++j) {
            auto prediction = nn.predict(testInputs[j]);
            bool predicted = prediction[0] > 0.5;
            bool expected = expectedOutputs[j][0] > 0.5;
            if (predicted == expected) correct++;
            
            std::cout << "    [" << testInputs[j][0] << ", " << testInputs[j][1] 
                      << "] -> " << std::fixed << std::setprecision(4) << prediction[0] 
                      << " (expected " << expectedOutputs[j][0] << ")" << std::endl;
        }
        
        double accuracy = static_cast<double>(correct) / testInputs.size();
        std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100 << "%" << std::endl;
    }
}

void demonstrateClassification() {
    std::cout << "\n========== BINARY CLASSIFICATION DEMO ==========\n" << std::endl;
    
    auto [features, labels] = generateClassificationDataset(1000);
    
    // Split into train and test (80/20)
    size_t trainSize = static_cast<size_t>(features.size() * 0.8);
    std::vector<std::vector<double>> trainX(features.begin(), features.begin() + trainSize);
    std::vector<std::vector<double>> trainY(labels.begin(), labels.begin() + trainSize);
    std::vector<std::vector<double>> testX(features.begin() + trainSize, features.end());
    std::vector<std::vector<double>> testY(labels.begin() + trainSize, labels.end());
    
    std::cout << "Dataset sizes - Train: " << trainX.size() << ", Test: " << testX.size() << std::endl;
    
    // Create and train network
    NeuralNetwork nn;
    nn.addLayer(std::make_unique<DenseLayer>(4, 16, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(16, 8, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(8, 1, ActivationType::SIGMOID));
    
    nn.compile(OptimizerType::ADAM, LossType::BINARY_CROSS_ENTROPY, 0.01);
    
    std::cout << "Training classification model..." << std::endl;
    nn.fit(trainX, trainY, 1000, 32);
    
    // Evaluate on test set
    std::vector<std::vector<double>> predictions;
    for (const auto& input : testX) {
        predictions.push_back(nn.predict(input));
    }
    
    // Calculate accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        bool predicted = predictions[i][0] > 0.5;
        bool expected = testY[i][0] > 0.5;
        if (predicted == expected) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / predictions.size();
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100 << "%" << std::endl;
    
    // Calculate MSE
    double mse = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double diff = predictions[i][0] - testY[i][0];
        mse += diff * diff;
    }
    mse /= predictions.size();
    std::cout << "Test MSE: " << std::fixed << std::setprecision(4) << mse << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "   C++ NEURAL NETWORK DEMO" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        // Demonstrate XOR learning with different optimizers
        demonstrateXORWithOptimizers();
        
        // Demonstrate binary classification
        demonstrateClassification();
        
        std::cout << "\n======================================" << std::endl;
        std::cout << "   ALL DEMOS COMPLETED SUCCESSFULLY!" << std::endl;
        std::cout << "======================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
