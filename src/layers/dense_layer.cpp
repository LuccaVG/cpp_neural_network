#include "dense_layer.h"
#include "initializer.h"
#include "activation.h"
#include <stdexcept>
#include <algorithm>

DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, ActivationType activation, InitializerType initializer)
    : inputSize(inputSize), outputSize(outputSize), activation(activation), initializer(initializer) {
    std::random_device rd;
    rng.seed(rd());
    weights = Initializer::initialize({inputSize, outputSize}, initializer, rng);
    biases.resize(outputSize, 0.0);
    weightGradients.resize(outputSize, std::vector<double>(inputSize, 0.0));
    biasGradients.resize(outputSize, 0.0);
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input, bool training) {
    if (input.size() != inputSize) {
        throw std::invalid_argument("Input size doesn't match layer input size");
    }

    lastInput = input;
    std::vector<double> output(outputSize, 0.0);
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            output[i] += weights[i][j] * input[j];
        }
        output[i] += biases[i];
    }

    lastOutput = output;
    return Activation::apply(output, activation);
}

std::vector<double> DenseLayer::backward(const std::vector<double>& outputGradient) {
    if (outputGradient.size() != outputSize) {
        throw std::invalid_argument("Output gradient size doesn't match layer output size");
    }

    std::vector<double> delta(outputSize);
    auto activationDerivative = Activation::derivative(lastOutput, activation);
    for (size_t i = 0; i < outputSize; ++i) {
        delta[i] = outputGradient[i] * activationDerivative[i];
    }

    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weightGradients[i][j] = delta[i] * lastInput[j];
        }
        biasGradients[i] = delta[i];
    }

    return delta; // Gradient with respect to the input
}

void DenseLayer::updateParameters(Optimizer& optimizer, int iteration) {
    for (size_t i = 0; i < outputSize; ++i) {
        optimizer.update(weights[i], weightGradients[i], iteration);
        biases[i] -= biasGradients[i]; // Update biases directly
    }
}

size_t DenseLayer::getParameterCount() const {
    return (inputSize * outputSize) + outputSize; // Weights + biases
}

size_t DenseLayer::getOutputSize() const {
    return outputSize;
}

LayerType DenseLayer::getType() const {
    return LayerType::DENSE;
}

std::string DenseLayer::getName() const {
    return "DenseLayer";
}

void DenseLayer::saveParameters(std::ofstream& file) const {
    // Save weights and biases to file
    for (const auto& row : weights) {
        for (const auto& w : row) {
            file << w << " ";
        }
        file << "\n";
    }
    for (const auto& b : biases) {
        file << b << " ";
    }
    file << "\n";
}

void DenseLayer::loadParameters(std::ifstream& file) {
    // Load weights and biases from file
    for (auto& row : weights) {
        for (auto& w : row) {
            file >> w;
        }
    }
    for (auto& b : biases) {
        file >> b;
    }
}

void DenseLayer::reset() {
    // Reset gradients
    for (auto& row : weightGradients) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(biasGradients.begin(), biasGradients.end(), 0.0);
}