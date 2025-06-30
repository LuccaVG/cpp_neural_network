#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include "enhanced_neural_network.h"
#include "layers/dense_layer.h"
#include "layers/dropout_layer.h"
#include "optimizers/advanced_optimizer.h"
#include "utils/dataset.h"
#include "utils/metrics.h"
#include "core/types.h"

// Generate synthetic classification dataset
Dataset generateClassificationDataset(int numSamples = 1000, int numFeatures = 4) {
    Dataset dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < numSamples; ++i) {
        std::vector<double> features(numFeatures);
        for (int j = 0; j < numFeatures; ++j) {
            features[j] = dist(gen);
        }
        
        // Create a non-linear classification boundary
        double decision = features[0] * features[1] + 0.5 * features[2] - 0.3 * features[3] + 0.2;
        std::vector<double> label = {decision > 0 ? 1.0 : 0.0};
        
        dataset.addSample(features, label);
    }
    
    return dataset;
}

// Generate XOR dataset for testing
Dataset generateXORDataset() {
    Dataset dataset;
    
    // XOR truth table - repeat for more training samples
    for (int repeat = 0; repeat < 250; ++repeat) {
        dataset.addSample({0, 0}, {0});
        dataset.addSample({0, 1}, {1});
        dataset.addSample({1, 0}, {1});
        dataset.addSample({1, 1}, {0});
    }
    
    return dataset;
}

void demonstrateBasicClassification() {
    std::cout << "\n========== BASIC CLASSIFICATION DEMO ==========\n" << std::endl;
    
    // Generate synthetic dataset
    auto dataset = generateClassificationDataset(1000, 4);
    dataset.normalize();
    dataset.shuffle();
    
    // Split into train and test sets
    auto [trainData, testData] = dataset.trainTestSplit(0.2); // 80% train, 20% test
    
    std::cout << "Dataset sizes:" << std::endl;
    std::cout << "  Training: " << trainData.size() << std::endl;
    std::cout << "  Test: " << testData.size() << std::endl;
    
    // Create enhanced neural network
    EnhancedNeuralNetwork nn;
    
    // Build architecture
    nn.addLayer(std::make_unique<DenseLayer>(4, 16, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(16, 8, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(8, 1, ActivationType::SIGMOID));
    
    // Compile with Adam optimizer
    nn.compile(OptimizerType::ADAM, LossType::BINARY_CROSS_ENTROPY, 0.01);
    
    // Train the model using the basic training method
    auto [trainX, trainY] = trainData.getTrainingData();
    auto [testX, testY] = testData.getTrainingData();
    
    std::cout << "Training classification model..." << std::endl;
    nn.fit(trainX, trainY, 1000, 32);
    
    // Evaluate on test set
    std::vector<std::vector<double>> predictions;
    for (const auto& input : testX) {
        predictions.push_back(nn.predict(input));
    }
    
    // Print evaluation metrics
    Metrics::printEvaluationSummary(predictions, testY);
}

void demonstrateXORWithDifferentOptimizers() {
    std::cout << "\n========== XOR WITH DIFFERENT OPTIMIZERS ==========\n" << std::endl;
    
    // Generate XOR dataset
    auto xorDataset = generateXORDataset();
    auto [trainX, trainY] = xorDataset.getTrainingData();
    
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
        
        EnhancedNeuralNetwork nn;
        nn.addLayer(std::make_unique<DenseLayer>(2, 8, ActivationType::RELU));
        nn.addLayer(std::make_unique<DenseLayer>(8, 4, ActivationType::RELU));
        nn.addLayer(std::make_unique<DenseLayer>(4, 1, ActivationType::SIGMOID));
        
        nn.compile(optimizers[i], LossType::BINARY_CROSS_ENTROPY, 0.01);
        
        // Train on XOR
        nn.fit(trainX, trainY, 2000, 4);
        
        // Test predictions
        std::cout << "XOR Results:" << std::endl;
        std::vector<std::vector<double>> testInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        std::vector<std::vector<double>> testOutputs = {{0}, {1}, {1}, {0}};
        
        std::vector<std::vector<double>> predictions;
        for (const auto& input : testInputs) {
            auto prediction = nn.predict(input);
            predictions.push_back(prediction);
            std::cout << "  [" << input[0] << ", " << input[1] << "] -> " 
                      << std::fixed << std::setprecision(4) << prediction[0] 
                      << " (expected: " << testOutputs[predictions.size()-1][0] << ")" << std::endl;
        }
        
        double accuracy = Metrics::accuracy(predictions, testOutputs);
        std::cout << "  Accuracy: " << std::fixed << std::setprecision(4) << accuracy << std::endl;
    }
}

void demonstrateActivationFunctions() {
    std::cout << "\n========== ACTIVATION FUNCTIONS DEMO ==========\n" << std::endl;
    
    std::vector<double> testValues = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::SIGMOID,
        ActivationType::TANH,
        ActivationType::LEAKY_RELU,
        ActivationType::ELU
    };
    
    std::vector<std::string> activationNames = {
        "ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "ELU"
    };
    
    for (size_t i = 0; i < activations.size(); ++i) {
        std::cout << "\n" << activationNames[i] << " activation:" << std::endl;
        for (double x : testValues) {
            double activated = Activation::apply(x, activations[i]);
            double derivative = Activation::derivative(x, activations[i]);
            std::cout << "  x=" << std::fixed << std::setprecision(2) << x 
                      << " -> f(x)=" << std::setprecision(4) << activated 
                      << ", f'(x)=" << std::setprecision(4) << derivative << std::endl;
        }
    }
}

void demonstrateAdvancedTraining() {
    std::cout << "\n========== ADVANCED TRAINING DEMO ==========\n" << std::endl;
    
    // Generate a larger dataset
    auto dataset = generateClassificationDataset(2000, 6);
    dataset.normalize();
    dataset.shuffle();
    
    // Split into train, validation, and test sets
    auto [tempData, testData] = dataset.trainTestSplit(0.2); // 80% temp, 20% test
    auto [trainData, valData] = tempData.trainTestSplit(0.25); // 60% train, 20% val
    
    std::cout << "Dataset sizes:" << std::endl;
    std::cout << "  Training: " << trainData.size() << std::endl;
    std::cout << "  Validation: " << valData.size() << std::endl;
    std::cout << "  Test: " << testData.size() << std::endl;
    
    // Create enhanced neural network with more complexity
    EnhancedNeuralNetwork nn;
    
    nn.addLayer(std::make_unique<DenseLayer>(6, 32, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(32, 16, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(16, 8, ActivationType::RELU));
    nn.addLayer(std::make_unique<DenseLayer>(8, 1, ActivationType::SIGMOID));
    
    nn.compile(OptimizerType::ADAM, LossType::BINARY_CROSS_ENTROPY, 0.001);
    
    // Enhanced training with validation
    auto [trainX, trainY] = trainData.getTrainingData();
    auto [valX, valY] = valData.getTrainingData();
    auto [testX, testY] = testData.getTrainingData();
    
    std::cout << "Training with validation monitoring..." << std::endl;
    nn.fitEnhanced(trainData, valData, 100, 32, true);
    
    // Final evaluation
    std::vector<std::vector<double>> predictions;
    for (const auto& input : testX) {
        predictions.push_back(nn.predict(input));
    }
    
    std::cout << "\nFinal Test Set Evaluation:" << std::endl;
    Metrics::printEvaluationSummary(predictions, testY);
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "   ENHANCED NEURAL NETWORK DEMO" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        // Demonstrate basic classification
        demonstrateBasicClassification();
        
        // Demonstrate XOR with different optimizers
        demonstrateXORWithDifferentOptimizers();
        
        // Demonstrate activation functions
        demonstrateActivationFunctions();
        
        // Demonstrate advanced training features
        demonstrateAdvancedTraining();
        
        std::cout << "\n========== DEMO COMPLETED SUCCESSFULLY ==========\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
