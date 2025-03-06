#include "batch_norm_layer.h"
#include "../core/initializer.h"  // Updated path
#include "../core/activation.h"   // Updated path
#include <cmath>
#include <stdexcept>
#include <numeric>  // Added for std::accumulate

BatchNormLayer::BatchNormLayer(size_t inputSize, double momentum)
    : inputSize(inputSize), momentum(momentum), runningMean(inputSize, 0.0), runningVariance(inputSize, 1.0) {
    gamma.resize(inputSize, 1.0);
    beta.resize(inputSize, 0.0);
    gradGamma.resize(inputSize, 0.0);
    gradBeta.resize(inputSize, 0.0);
}

std::vector<double> BatchNormLayer::forward(const std::vector<double>& input, bool training) {
    if (input.size() != inputSize) {
        throw std::invalid_argument("Input size does not match layer input size");
    }

    if (training) {
        // Calculate mean and variance
        double mean = std::accumulate(input.begin(), input.end(), 0.0) / input.size();
        double variance = 0.0;
        for (const auto& value : input) {
            variance += (value - mean) * (value - mean);
        }
        variance /= input.size();

        // Update running mean and variance
        runningMean = momentum * runningMean + (1.0 - momentum) * mean;
        runningVariance = momentum * runningVariance + (1.0 - momentum) * variance;

        // Normalize the input
        std::vector<double> normalized(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            normalized[i] = (input[i] - mean) / std::sqrt(variance + 1e-8);
        }

        // Scale and shift
        for (size_t i = 0; i < normalized.size(); ++i) {
            lastOutput[i] = gamma[i] * normalized[i] + beta[i];
        }
    } else {
        // Use running mean and variance for inference
        std::vector<double> normalized(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            normalized[i] = (input[i] - runningMean[i]) / std::sqrt(runningVariance[i] + 1e-8);
            lastOutput[i] = gamma[i] * normalized[i] + beta[i];
        }
    }

    return lastOutput;
}

std::vector<double> BatchNormLayer::backward(const std::vector<double>& outputGradient) {
    std::vector<double> inputGradient(inputSize, 0.0);
    std::vector<double> normalized(inputSize, 0.0);

    // Calculate gradients for gamma and beta
    for (size_t i = 0; i < outputGradient.size(); ++i) {
        gradGamma[i] += outputGradient[i] * lastOutput[i];
        gradBeta[i] += outputGradient[i];
    }

    // Calculate input gradient
    for (size_t i = 0; i < outputGradient.size(); ++i) {
        inputGradient[i] = (outputGradient[i] - gradBeta[i]) * gamma[i] / std::sqrt(runningVariance[i] + 1e-8);
    }

    return inputGradient;
}

void BatchNormLayer::updateParameters(Optimizer& optimizer, int iteration) {
    optimizer.update(gamma, gradGamma, iteration);
    optimizer.update(beta, gradBeta, iteration);
    std::fill(gradGamma.begin(), gradGamma.end(), 0.0);
    std::fill(gradBeta.begin(), gradBeta.end(), 0.0);
}

size_t BatchNormLayer::getParameterCount() const {
    return gamma.size() + beta.size();
}

size_t BatchNormLayer::getOutputSize() const {
    return inputSize;
}

LayerType BatchNormLayer::getType() const {
    return LayerType::BATCH_NORMALIZATION;
}

std::string BatchNormLayer::getName() const {
    return "BatchNormLayer";
}

void BatchNormLayer::saveParameters(std::ofstream& file) const {
    // Save gamma and beta parameters to file
    for (const auto& g : gamma) {
        file << g << " ";
    }
    file << std::endl;
    for (const auto& b : beta) {
        file << b << " ";
    }
    file << std::endl;
}

void BatchNormLayer::loadParameters(std::ifstream& file) {
    // Load gamma and beta parameters from file
    for (auto& g : gamma) {
        file >> g;
    }
    for (auto& b : beta) {
        file >> b;
    }
}

void BatchNormLayer::reset() {
    // Reset gradients
    std::fill(gradGamma.begin(), gradGamma.end(), 0.0);
    std::fill(gradBeta.begin(), gradBeta.end(), 0.0);
}