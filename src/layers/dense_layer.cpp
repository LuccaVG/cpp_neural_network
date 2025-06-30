#include "dense_layer.h"
#include "../core/activation.h"
#include <stdexcept>
#include <random>
#include <fstream>

DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, ActivationType activation)
    : inputSize(inputSize), outputSize(outputSize), activation(activation) {
    
    // Initialize weights with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / inputSize)); // He initialization for ReLU
    
    weights.resize(outputSize, std::vector<double>(inputSize));
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weights[i][j] = dist(gen);
        }
    }
    
    // Initialize biases to zero
    biases.resize(outputSize, 0.0);
    
    // Initialize storage vectors
    lastInput.resize(inputSize);
    lastOutput.resize(outputSize);
    lastActivatedOutput.resize(outputSize);
    inputGradient.resize(inputSize);
    lastDelta.resize(outputSize);
}

void DenseLayer::forward(const std::vector<double>& input) {
    if (input.size() != inputSize) {
        throw std::invalid_argument("Input size doesn't match layer input size");
    }

    lastInput = input;
    
    // Compute linear transformation: output = weights * input + bias
    for (size_t i = 0; i < outputSize; ++i) {
        lastOutput[i] = biases[i];
        for (size_t j = 0; j < inputSize; ++j) {
            lastOutput[i] += weights[i][j] * input[j];
        }
    }

    // Apply activation function
    lastActivatedOutput = Activation::apply(lastOutput, activation);
}

std::vector<double> DenseLayer::backward(const std::vector<double>& outputGradient) {
    if (outputGradient.size() != outputSize) {
        throw std::invalid_argument("Output gradient size doesn't match layer output size");
    }

    // Compute gradient of activation function
    std::vector<double> activationGrad = Activation::derivative(lastOutput, activation);
    
    // Element-wise multiply with output gradient (delta = gradient * activation_derivative)
    std::vector<double> delta(outputSize);
    for (size_t i = 0; i < outputSize; ++i) {
        delta[i] = outputGradient[i] * activationGrad[i];
    }

    // Compute input gradient (for backpropagation to previous layer)
    std::fill(inputGradient.begin(), inputGradient.end(), 0.0);
    for (size_t j = 0; j < inputSize; ++j) {
        for (size_t i = 0; i < outputSize; ++i) {
            inputGradient[j] += delta[i] * weights[i][j];
        }
    }

    // Store delta for parameter updates
    lastDelta = delta;

    return inputGradient; // Return input gradient for previous layer
}

void DenseLayer::updateParameters(Optimizer& optimizer, int iteration) {
    // Compute weight gradients
    std::vector<double> weightGradients;
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weightGradients.push_back(lastDelta[i] * lastInput[j]);
        }
    }
    
    // Flatten weights for optimizer update
    std::vector<double> flatWeights;
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            flatWeights.push_back(weights[i][j]);
        }
    }
    
    // Update weights using optimizer
    optimizer.update(flatWeights, weightGradients, 0.01); // Learning rate
    
    // Unflatten weights
    size_t idx = 0;
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weights[i][j] = flatWeights[idx++];
        }
    }
    
    // Update biases using optimizer
    optimizer.update(biases, lastDelta, 0.01); // Learning rate
}

std::vector<double> DenseLayer::getOutput() const {
    return lastActivatedOutput;
}

std::vector<double> DenseLayer::getInputGradient() const {
    return inputGradient;
}

void DenseLayer::save(std::ofstream& file) const {
    // Save layer dimensions
    file.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
    file.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));
    file.write(reinterpret_cast<const char*>(&activation), sizeof(activation));
    
    // Save weights
    for (const auto& row : weights) {
        for (double w : row) {
            file.write(reinterpret_cast<const char*>(&w), sizeof(w));
        }
    }
    
    // Save biases
    for (double b : biases) {
        file.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }
}

void DenseLayer::load(std::ifstream& file) {
    // Load layer dimensions
    file.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    file.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
    file.read(reinterpret_cast<char*>(&activation), sizeof(activation));
    
    // Resize vectors
    weights.resize(outputSize, std::vector<double>(inputSize));
    biases.resize(outputSize);
    lastInput.resize(inputSize);
    lastOutput.resize(outputSize);
    lastActivatedOutput.resize(outputSize);
    inputGradient.resize(inputSize);
    
    // Load weights
    for (auto& row : weights) {
        for (double& w : row) {
            file.read(reinterpret_cast<char*>(&w), sizeof(w));
        }
    }
    
    // Load biases
    for (double& b : biases) {
        file.read(reinterpret_cast<char*>(&b), sizeof(b));
    }
}
