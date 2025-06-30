#ifndef ENHANCED_NEURAL_NETWORK_H
#define ENHANCED_NEURAL_NETWORK_H

#include "neural_network.h"
#include "utils/dataset.h"
#include "utils/metrics.h"
#include "optimizers/advanced_optimizer.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>

class EnhancedNeuralNetwork : public NeuralNetwork {
private:
    // Training history
    std::vector<double> trainLossHistory;
    std::vector<double> valLossHistory;
    std::vector<double> trainAccuracyHistory;
    std::vector<double> valAccuracyHistory;
    
    // Early stopping parameters
    bool useEarlyStopping = false;
    double bestValLoss = std::numeric_limits<double>::max();
    int patienceCounter = 0;
    int patience = 10;
    
    // Training mode
    bool isTraining = true;
    
public:
    EnhancedNeuralNetwork() : NeuralNetwork() {}
    
    // Enhanced training with validation and early stopping
    void fitEnhanced(const Dataset& trainData, const Dataset& valData, 
                    int epochs, int batchSize = 32, bool verbose = true) {
        if (layers.empty()) {
            throw std::runtime_error("No layers added to the network");
        }
        
        if (!optimizer || !loss) {
            throw std::runtime_error("Network not compiled");
        }
        
        auto [trainX, trainY] = trainData.getTrainingData();
        auto [valX, valY] = valData.getTrainingData();
        
        trainLossHistory.clear();
        valLossHistory.clear();
        trainAccuracyHistory.clear();
        valAccuracyHistory.clear();
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Training phase
            setTraining(true);
            double trainLoss = 0.0;
            
            // Shuffle training data
            std::vector<size_t> indices(trainX.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            
            // Mini-batch training
            for (size_t i = 0; i < trainX.size(); i += batchSize) {
                double batchLoss = 0.0;
                
                for (size_t j = i; j < std::min(i + batchSize, trainX.size()); ++j) {
                    size_t idx = indices[j];
                    
                    // Forward pass
                    forward(trainX[idx]);
                    std::vector<double> output = layers.back()->getOutput();
                    batchLoss += loss->calculate(output, trainY[idx]);
                    
                    // Backward pass
                    std::vector<double> lossGradient = loss->calculateGradient(output, trainY[idx]);
                    backward(lossGradient);
                    
                    // Update parameters
                    updateParameters(epoch * trainX.size() + j);
                }
                
                trainLoss += batchLoss;
            }
            
            trainLoss /= trainX.size();
            trainLossHistory.push_back(trainLoss);
            
            // Validation phase
            setTraining(false);
            double valLoss = 0.0;
            std::vector<std::vector<double>> trainPreds, valPreds;
            
            // Calculate training accuracy
            for (const auto& sample : trainX) {
                trainPreds.push_back(predict(sample));
            }
            double trainAcc = Metrics::accuracy(trainPreds, trainY);
            trainAccuracyHistory.push_back(trainAcc);
            
            // Calculate validation loss and accuracy
            for (size_t i = 0; i < valX.size(); ++i) {
                std::vector<double> pred = predict(valX[i]);
                valPreds.push_back(pred);
                valLoss += loss->calculate(pred, valY[i]);
            }
            valLoss /= valX.size();
            valLossHistory.push_back(valLoss);
            
            double valAcc = Metrics::accuracy(valPreds, valY);
            valAccuracyHistory.push_back(valAcc);
            
            // Print progress
            if (verbose && epoch % 100 == 0) {
                std::cout << "Epoch " << epoch 
                         << " - Train Loss: " << trainLoss 
                         << " - Train Acc: " << trainAcc
                         << " - Val Loss: " << valLoss 
                         << " - Val Acc: " << valAcc << std::endl;
            }
            
            // Early stopping check
            if (useEarlyStopping) {
                if (valLoss < bestValLoss) {
                    bestValLoss = valLoss;
                    patienceCounter = 0;
                    // Save best model weights here
                } else {
                    patienceCounter++;
                    if (patienceCounter >= patience) {
                        std::cout << "Early stopping at epoch " << epoch << std::endl;
                        break;
                    }
                }
            }
        }
    }
    
    // Set training mode for layers like dropout
    void setTraining(bool training) {
        isTraining = training;
        // This would need to be implemented in layers that have different behavior during training/inference
    }
    
    // Enable early stopping
    void enableEarlyStopping(int patienceEpochs = 10) {
        useEarlyStopping = true;
        patience = patienceEpochs;
    }
    
    // Model evaluation
    void evaluate(const Dataset& testData, bool verbose = true) {
        auto [testX, testY] = testData.getTrainingData();
        
        setTraining(false);
        std::vector<std::vector<double>> predictions;
        double totalLoss = 0.0;
        
        for (size_t i = 0; i < testX.size(); ++i) {
            std::vector<double> pred = predict(testX[i]);
            predictions.push_back(pred);
            totalLoss += loss->calculate(pred, testY[i]);
        }
        
        totalLoss /= testX.size();
        double accuracy = Metrics::accuracy(predictions, testY);
        double mse = Metrics::meanSquaredError(predictions, testY);
        
        if (verbose) {
            std::cout << "\n=== Model Evaluation ===" << std::endl;
            std::cout << "Test Loss: " << totalLoss << std::endl;
            std::cout << "Test Accuracy: " << accuracy << std::endl;
            std::cout << "Test MSE: " << mse << std::endl;
            
            // Print confusion matrix for binary classification
            if (testY[0].size() == 1) {
                auto cm = Metrics::confusionMatrix(predictions, testY);
                cm.print();
            }
        }
    }
    
    // Get training history
    const std::vector<double>& getTrainLossHistory() const { return trainLossHistory; }
    const std::vector<double>& getValLossHistory() const { return valLossHistory; }
    const std::vector<double>& getTrainAccuracyHistory() const { return trainAccuracyHistory; }
    const std::vector<double>& getValAccuracyHistory() const { return valAccuracyHistory; }
    
    // Save training history to file
    void saveTrainingHistory(const std::string& filename) const {
        std::ofstream file(filename);
        file << "Epoch,TrainLoss,ValLoss,TrainAcc,ValAcc\n";
        
        for (size_t i = 0; i < trainLossHistory.size(); ++i) {
            file << i << "," 
                 << trainLossHistory[i] << "," 
                 << (i < valLossHistory.size() ? valLossHistory[i] : 0.0) << ","
                 << (i < trainAccuracyHistory.size() ? trainAccuracyHistory[i] : 0.0) << ","
                 << (i < valAccuracyHistory.size() ? valAccuracyHistory[i] : 0.0) << "\n";
        }
    }
};

#endif // ENHANCED_NEURAL_NETWORK_H
