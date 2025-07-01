#include "batch_norm_layer.h"
#include <cmath>
#include <stdexcept>
#include <numeric>  // Added for std::accumulate

BatchNormLayer::BatchNormLayer(size_t inputSize, double momentum, double epsilon)
    : inputSize(inputSize), momentum(momentum), epsilon(epsilon), 
      runningMean(inputSize, 0.0), runningVariance(inputSize, 1.0) {
    gamma.resize(inputSize, 1.0);
    beta.resize(inputSize, 0.0);
    gradGamma.resize(inputSize, 0.0);
    gradBeta.resize(inputSize, 0.0);
    lastOutput.resize(inputSize, 0.0);
    inputGradient.resize(inputSize, 0.0);
}

void BatchNormLayer::forward(const std::vector<double>& input) {
    if (input.size() != inputSize) {
        throw std::invalid_argument("Input size does not match layer input size");
    }

    lastInput = input;
    
    if (isTraining) {
        // Calculate mean and variance
        double mean = std::accumulate(input.begin(), input.end(), 0.0) / input.size();
        double variance = 0.0;
        for (const auto& value : input) {
            variance += (value - mean) * (value - mean);
        }
        variance /= input.size();

        // Update running mean and variance (element-wise for proper batching)
        for (size_t i = 0; i < inputSize; ++i) {
            runningMean[i] = momentum * runningMean[i] + (1.0 - momentum) * mean;
            runningVariance[i] = momentum * runningVariance[i] + (1.0 - momentum) * variance;
        }

        // Normalize the input
        for (size_t i = 0; i < input.size(); ++i) {
            double normalized = (input[i] - mean) / std::sqrt(variance + epsilon);
            lastOutput[i] = gamma[i] * normalized + beta[i];
        }
    } else {
        // Use running mean and variance for inference
        for (size_t i = 0; i < input.size(); ++i) {
            double normalized = (input[i] - runningMean[i]) / std::sqrt(runningVariance[i] + epsilon);
            lastOutput[i] = gamma[i] * normalized + beta[i];
        }
    }
}

std::vector<double> BatchNormLayer::backward(const std::vector<double>& outputGradient) {
    if (outputGradient.size() != inputSize) {
        throw std::invalid_argument("Output gradient size does not match layer output size");
    }

    // Simple implementation for now - proper batch norm backward pass is complex
    // This is a simplified version that should work for basic training
    for (size_t i = 0; i < outputGradient.size(); ++i) {
        gradGamma[i] += outputGradient[i] * lastOutput[i];
        gradBeta[i] += outputGradient[i];
        inputGradient[i] = outputGradient[i] * gamma[i];
    }

    return inputGradient;
}

void BatchNormLayer::updateParameters(Optimizer& optimizer, int iteration) {
    // Combine gamma and beta into single parameter vector to maintain optimizer state consistency
    std::vector<double> allParams;
    std::vector<double> allGradients;
    
    // Add gamma parameters and gradients
    allParams.insert(allParams.end(), gamma.begin(), gamma.end());
    allGradients.insert(allGradients.end(), gradGamma.begin(), gradGamma.end());
    
    // Add beta parameters and gradients
    allParams.insert(allParams.end(), beta.begin(), beta.end());
    allGradients.insert(allGradients.end(), gradBeta.begin(), gradBeta.end());
    
    // Single optimizer update call
    double learningRate = 0.01;
    optimizer.update(allParams, allGradients, learningRate);
    
    // Extract updated gamma parameters
    for (size_t i = 0; i < gamma.size(); ++i) {
        gamma[i] = allParams[i];
    }
    
    // Extract updated beta parameters
    for (size_t i = 0; i < beta.size(); ++i) {
        beta[i] = allParams[gamma.size() + i];
    }
    
    // Clear gradients
    std::fill(gradGamma.begin(), gradGamma.end(), 0.0);
    std::fill(gradBeta.begin(), gradBeta.end(), 0.0);
}

std::vector<double> BatchNormLayer::getOutput() const {
    return lastOutput;
}

std::vector<double> BatchNormLayer::getInputGradient() const {
    return inputGradient;
}

void BatchNormLayer::save(std::ofstream& file) const {
    // Save layer parameters
    file.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
    file.write(reinterpret_cast<const char*>(&momentum), sizeof(momentum));
    file.write(reinterpret_cast<const char*>(&epsilon), sizeof(epsilon));
    
    // Save gamma and beta
    for (const auto& g : gamma) {
        file.write(reinterpret_cast<const char*>(&g), sizeof(g));
    }
    for (const auto& b : beta) {
        file.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }
    
    // Save running statistics
    for (const auto& m : runningMean) {
        file.write(reinterpret_cast<const char*>(&m), sizeof(m));
    }
    for (const auto& v : runningVariance) {
        file.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
}

void BatchNormLayer::load(std::ifstream& file) {
    // Load layer parameters
    file.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
    file.read(reinterpret_cast<char*>(&momentum), sizeof(momentum));
    file.read(reinterpret_cast<char*>(&epsilon), sizeof(epsilon));
    
    // Resize vectors
    gamma.resize(inputSize);
    beta.resize(inputSize);
    runningMean.resize(inputSize);
    runningVariance.resize(inputSize);
    gradGamma.resize(inputSize, 0.0);
    gradBeta.resize(inputSize, 0.0);
    lastOutput.resize(inputSize, 0.0);
    inputGradient.resize(inputSize, 0.0);
    
    // Load gamma and beta
    for (auto& g : gamma) {
        file.read(reinterpret_cast<char*>(&g), sizeof(g));
    }
    for (auto& b : beta) {
        file.read(reinterpret_cast<char*>(&b), sizeof(b));
    }
    
    // Load running statistics
    for (auto& m : runningMean) {
        file.read(reinterpret_cast<char*>(&m), sizeof(m));
    }
    for (auto& v : runningVariance) {
        file.read(reinterpret_cast<char*>(&v), sizeof(v));
    }
}