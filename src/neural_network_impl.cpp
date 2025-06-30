#include "neural_network.h"
#include "optimizers/sgd.h"
#include "optimizers/adam.h"
#include "optimizers/momentum.h"
#include "optimizers/rmsprop.h"
#include "core/loss.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

NeuralNetwork::NeuralNetwork() : optimizer(nullptr), loss(nullptr), lossType(LossType::MEAN_SQUARED_ERROR) {
}

NeuralNetwork::~NeuralNetwork() {
}

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

void NeuralNetwork::compile(OptimizerType optimizerType, LossType lossType, double learningRate) {
    this->lossType = lossType;
    
    // Create optimizer based on type
    switch (optimizerType) {
        case OptimizerType::SGD:
            optimizer = std::make_unique<SGD>();
            break;
        case OptimizerType::ADAM:
            optimizer = std::make_unique<Adam>();
            break;
        case OptimizerType::MOMENTUM:
            optimizer = std::make_unique<Momentum>();
            break;
        case OptimizerType::RMSPROP:
            optimizer = std::make_unique<RMSProp>();
            break;
        default:
            throw std::runtime_error("Unsupported optimizer type");
    }
    
    // Create loss function
    loss = Loss::create(lossType);
}

void NeuralNetwork::fit(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, int epochs, int batchSize) {
    if (layers.empty()) {
        throw std::runtime_error("No layers added to the network");
    }
    
    if (!optimizer || !loss) {
        throw std::runtime_error("Network not compiled");
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;
        
        for (size_t i = 0; i < x.size(); ++i) {
            // Forward pass
            forward(x[i]);
            
            // Calculate loss
            std::vector<double> output = layers.back()->getOutput();
            totalLoss += loss->calculate(output, y[i]);
            
            // Backward pass
            std::vector<double> lossGradient = loss->calculateGradient(output, y[i]);
            backward(lossGradient);
            
            // Update parameters
            updateParameters(epoch * x.size() + i);
        }
        
        // Print loss every 1000 epochs
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / x.size() << std::endl;
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    forward(input);
    return layers.back()->getOutput();
}

void NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> currentInput = input;
    
    for (auto& layer : layers) {
        layer->forward(currentInput);
        currentInput = layer->getOutput();
    }
}

void NeuralNetwork::backward(const std::vector<double>& outputGradient) {
    std::vector<double> currentGradient = outputGradient;
    
    for (int i = layers.size() - 1; i >= 0; --i) {
        currentGradient = layers[i]->backward(currentGradient);
    }
}

void NeuralNetwork::updateParameters(int iteration) {
    for (auto& layer : layers) {
        layer->updateParameters(*optimizer, iteration);
    }
}

void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for saving: " + filename);
    }
    
    // Save network metadata
    size_t numLayers = layers.size();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
    file.write(reinterpret_cast<const char*>(&lossType), sizeof(lossType));
    
    // Save each layer
    for (const auto& layer : layers) {
        layer->save(file);
    }
    
    file.close();
}

void NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for loading: " + filename);
    }
    
    // Clear existing layers
    layers.clear();
    
    // Load network metadata
    size_t numLayers;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
    file.read(reinterpret_cast<char*>(&lossType), sizeof(lossType));
    
    // Load each layer
    for (size_t i = 0; i < numLayers; ++i) {
        // Note: This is simplified - in a full implementation, you'd need
        // to save/load layer type information to recreate the correct layer type
        // For now, this is a placeholder that would need proper implementation
        
        // auto layer = Layer::create(layerType);
        // layer->load(file);
        // layers.push_back(std::move(layer));
    }
    
    file.close();
    
    // Recreate loss function
    loss = Loss::create(lossType);
}
