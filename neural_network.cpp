#include "neural_network.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>

// Constructor with enhanced initialization
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                           ActivationFunction hiddenActivation,
                           ActivationFunction outputActivation,
                           InitializationMethod initMethod)
    : layerSizes(layerSizes),
      hiddenActivation(hiddenActivation),
      outputActivation(outputActivation) {
    
    // Use a random device to seed the random number generator
    std::random_device rd;
    rng = std::mt19937(rd());
    
    // Initialize weights using the specified method
    initializeWeights(initMethod);
    
    // Initialize optimizer state variables
    weightVelocities.resize(weights.size());
    biasVelocities.resize(biases.size());
    weightMoments1.resize(weights.size());
    biasMoments1.resize(biases.size());
    weightMoments2.resize(weights.size());
    biasMoments2.resize(biases.size());
    
    // Initialize batch normalization parameters
    batchNormLayers.resize(layerSizes.size() - 1);
    
    for (size_t i = 0; i < weights.size(); i++) {
        weightVelocities[i].resize(weights[i].size());
        weightMoments1[i].resize(weights[i].size());
        weightMoments2[i].resize(weights[i].size());
        
        for (size_t j = 0; j < weights[i].size(); j++) {
            weightVelocities[i][j].resize(weights[i][j].size(), 0.0);
            weightMoments1[i][j].resize(weights[i][j].size(), 0.0);
            weightMoments2[i][j].resize(weights[i][j].size(), 0.0);
        }
        
        biasVelocities[i].resize(biases[i].size(), 0.0);
        biasMoments1[i].resize(biases[i].size(), 0.0);
        biasMoments2[i].resize(biases[i].size(), 0.0);
        
        // Initialize batch normalization parameters for this layer
        batchNormLayers[i].gamma.resize(layerSizes[i + 1], 1.0);
        batchNormLayers[i].beta.resize(layerSizes[i + 1], 0.0);
        batchNormLayers[i].movingMean.resize(layerSizes[i + 1], 0.0);
        batchNormLayers[i].movingVar.resize(layerSizes[i + 1], 1.0);
    }
}

void NeuralNetwork::initializeWeights(InitializationMethod method) {
    weights.clear();
    biases.clear();
    
    // Create distributions for different initialization methods
    std::uniform_real_distribution<double> uniformDist(-0.5, 0.5);
    std::normal_distribution<double> normalDist(0.0, 1.0);
    
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        weights.push_back(std::vector<std::vector<double>>(
            layerSizes[i], std::vector<double>(layerSizes[i-1])));
        biases.push_back(std::vector<double>(layerSizes[i], 0.0));
        
        double scale = 1.0;
        
        switch (method) {
            case InitializationMethod::XAVIER:
                // Xavier/Glorot initialization
                scale = sqrt(6.0 / (layerSizes[i-1] + layerSizes[i]));
                break;
                
            case InitializationMethod::HE:
                // He initialization (better for ReLU)
                scale = sqrt(2.0 / layerSizes[i-1]);
                break;
                
            case InitializationMethod::LECUN:
                // LeCun initialization
                scale = sqrt(1.0 / layerSizes[i-1]);
                break;
                
            case InitializationMethod::NORMAL:
                // Normal distribution
                scale = 0.1;
                break;
                
            case InitializationMethod::UNIFORM:
                // Uniform distribution
                scale = 0.5;
                break;
        }
        
        // Initialize weights based on the selected method
        for (auto& neuron : weights.back()) {
            for (auto& weight : neuron) {
                if (method == InitializationMethod::NORMAL) {
                    weight = normalDist(rng) * scale;
                } else {
                    weight = uniformDist(rng) * scale;
                }
            }
        }
        
        // Initialize biases
        for (auto& bias : biases.back()) {
            if (method == InitializationMethod::NORMAL) {
                bias = normalDist(rng) * 0.1;
            } else {
                bias = 0.0; // Most modern networks initialize biases to zero
            }
        }
    }
}

std::vector<std::vector<double>> NeuralNetwork::forwardPass(const std::vector<double>& input) const {
    std::vector<std::vector<double>> activations;
    activations.push_back(input);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(layerSizes[i+1], 0.0);
        
        // Calculate weighted sum for each neuron
        for (size_t j = 0; j < weights[i].size(); ++j) {
            z[j] = biases[i][j];
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                z[j] += weights[i][j][k] * activations.back()[k];
            }
        }
        
        // Apply activation function
        ActivationFunction activation = (i == weights.size() - 1) ? outputActivation : hiddenActivation;
        activations.push_back(applyActivation(z, activation));
    }
    
    return activations;
}

std::vector<std::vector<double>> NeuralNetwork::forwardPassWithDropout(
    const std::vector<double>& input, 
    double dropoutRate,
    std::vector<std::vector<bool>>& dropoutMasks) {
    
    std::vector<std::vector<double>> activations;
    activations.push_back(input);
    
    dropoutMasks.clear();
    std::uniform_real_distribution<double> dropoutDist(0.0, 1.0);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(layerSizes[i+1], 0.0);
        
        // Calculate weighted sum for each neuron
        for (size_t j = 0; j < weights[i].size(); ++j) {
            z[j] = biases[i][j];
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                z[j] += weights[i][j][k] * activations.back()[k];
            }
        }
        
        // Apply activation function
        ActivationFunction activation = (i == weights.size() - 1) ? outputActivation : hiddenActivation;
        std::vector<double> layerActivation = applyActivation(z, activation);
        
        // Apply dropout (except for output layer)
        if (i < weights.size() - 1 && dropoutRate > 0) {
            std::vector<bool> dropoutMask(layerActivation.size(), false);
            
            for (size_t j = 0; j < layerActivation.size(); ++j) {
                // Keep neuron with probability (1-dropoutRate)
                if (dropoutDist(rng) >= dropoutRate) {
                    dropoutMask[j] = true;
                    // Scale by 1/(1-dropoutRate) to maintain expected value
                    layerActivation[j] /= (1.0 - dropoutRate);
                } else {
                    layerActivation[j] = 0.0;
                }
            }
            
            dropoutMasks.push_back(dropoutMask);
        }
        
        activations.push_back(layerActivation);
    }
    
    return activations;
}

std::vector<double> NeuralNetwork::applyActivation(const std::vector<double>& z, ActivationFunction activation) const {
    std::vector<double> result(z.size());
    
    switch (activation) {
        case ActivationFunction::SIGMOID:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = 1.0 / (1.0 + std::exp(-z[i]));
            }
            break;
            
        case ActivationFunction::TANH:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = std::tanh(z[i]);
            }
            break;
            
        case ActivationFunction::RELU:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = std::max(0.0, z[i]);
            }
            break;
            
        case ActivationFunction::LEAKY_RELU:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = z[i] > 0 ? z[i] : 0.01 * z[i];
            }
            break;
            
        case ActivationFunction::ELU:
            result = elu(z);
            break;
            
        case ActivationFunction::SWISH:
            result = swish(z);
            break;
            
        case ActivationFunction::MISH:
            result = mish(z);
            break;
            
        case ActivationFunction::SOFTMAX:
            result = softmax(z);
            break;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::applyActivationDerivative(const std::vector<double>& z, ActivationFunction activation) const {
    std::vector<double> result(z.size());
    std::vector<double> activated = applyActivation(z, activation);
    
    switch (activation) {
        case ActivationFunction::SIGMOID:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = activated[i] * (1.0 - activated[i]);
            }
            break;
            
        case ActivationFunction::TANH:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = 1.0 - activated[i] * activated[i];
            }
            break;
            
        case ActivationFunction::RELU:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = z[i] > 0 ? 1.0 : 0.0;
            }
            break;
            
        case ActivationFunction::LEAKY_RELU:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = z[i] > 0 ? 1.0 : 0.01;
            }
            break;
            
        case ActivationFunction::ELU:
            for (size_t i = 0; i < z.size(); ++i) {
                if (z[i] > 0) {
                    result[i] = 1.0;
                } else {
                    // ELU derivative: alpha * exp(x) for x < 0
                    double alpha = 1.0;
                    result[i] = alpha * std::exp(z[i]);
                }
            }
            break;
            
        case ActivationFunction::SWISH:
            for (size_t i = 0; i < z.size(); ++i) {
                double sigmoid = 1.0 / (1.0 + std::exp(-z[i]));
                // Swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                result[i] = sigmoid + z[i] * sigmoid * (1.0 - sigmoid);
            }
            break;
            
        case ActivationFunction::MISH:
            for (size_t i = 0; i < z.size(); ++i) {
                // Mish derivative is complex; this is a simplified version
                double exp_x = std::exp(z[i]);
                double exp_2x = std::exp(2 * z[i]);
                double exp_3x = std::exp(3 * z[i]);
                double omega = 4.0 * (z[i] + 1.0) + 4.0 * exp_2x + exp_3x + exp_x * (4.0 * z[i] + 6.0);
                double delta = 2.0 * exp_x + exp_2x + 2.0;
                result[i] = exp_x * omega / (delta * delta);
            }
            break;
            
        case ActivationFunction::SOFTMAX:
            // For softmax, we typically handle the derivative differently in backpropagation
            // because we often use cross-entropy loss which simplifies the gradient
            for (size_t i = 0; i < z.size(); ++i) {
                // This is a placeholder; actual softmax derivative is a Jacobian matrix
                result[i] = activated[i] * (1.0 - activated[i]);
            }
            break;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    
    // Find max value for numerical stability
    double max_val = *std::max_element(z.begin(), z.end());
    
    // Compute exp(z_i - max_val) and sum
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        result[i] = std::exp(z[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    for (size_t i = 0; i < z.size(); ++i) {
        result[i] /= sum;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::elu(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    double alpha = 1.0; // Parameter controlling the value for negative inputs
    
    for (size_t i = 0; i < z.size(); ++i) {
        if (z[i] > 0) {
            result[i] = z[i];
        } else {
            result[i] = alpha * (std::exp(z[i]) - 1.0);
        }
    }
    
    return result;
}

std::vector<double> NeuralNetwork::swish(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    
    for (size_t i = 0; i < z.size(); ++i) {
        double sigmoid = 1.0 / (1.0 + std::exp(-z[i]));
        result[i] = z[i] * sigmoid;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::mish(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    
    for (size_t i = 0; i < z.size(); ++i) {
        // Softplus: log(1 + exp(x))
        double softplus = std::log1p(std::exp(z[i]));
        // Mish: x * tanh(softplus(x))
        result[i] = z[i] * std::tanh(softplus);
    }
    
    return result;
}

std::vector<double> NeuralNetwork::batchNormalize(const std::vector<double>& input, 
                                               size_t layerIndex,
                                               bool isTraining) {
    std::vector<double> result(input.size());
    const double epsilon = 1e-8;
    const double momentum = 0.9;
    
    if (isTraining) {
        // Calculate mean
        double mean = 0.0;
        for (size_t i = 0; i < input.size(); ++i) {
            mean += input[i];
        }
        mean /= input.size();
        
        // Calculate variance
        double variance = 0.0;
        for (size_t i = 0; i < input.size(); ++i) {
            double diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= input.size();
        
        // Update moving statistics
        for (size_t i = 0; i < input.size(); ++i) {
            batchNormLayers[layerIndex].movingMean[i] = 
                batchNormLayers[layerIndex].movingMean[i] * momentum + mean * (1 - momentum);
            batchNormLayers[layerIndex].movingVar[i] = 
                batchNormLayers[layerIndex].movingVar[i] * momentum + variance * (1 - momentum);
        }
        
        // Normalize
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = (input[i] - mean) / std::sqrt(variance + epsilon);
            // Scale and shift
            result[i] = result[i] * batchNormLayers[layerIndex].gamma[i] + batchNormLayers[layerIndex].beta[i];
        }
    } else {
        // Use stored mean and variance for inference
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = (input[i] - batchNormLayers[layerIndex].movingMean[i]) / 
                std::sqrt(batchNormLayers[layerIndex].movingVar[i] + epsilon);
            // Scale and shift
            result[i] = result[i] * batchNormLayers[layerIndex].gamma[i] + batchNormLayers[layerIndex].beta[i];
        }
    }
    
    return result;
}

void NeuralNetwork::applyAdamOptimizer(std::vector<std::vector<std::vector<double>>>& weightGradients,
                                     std::vector<std::vector<double>>& biasGradients,
                                     double learningRate, double beta1, double beta2, 
                                     double epsilon, int timestep) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                // Update biased first moment estimate
                weightMoments1[i][j][k] = beta1 * weightMoments1[i][j][k] + 
                                        (1 - beta1) * weightGradients[i][j][k];
                
                // Update biased second raw moment estimate
                weightMoments2[i][j][k] = beta2 * weightMoments2[i][j][k] + 
                                        (1 - beta2) * weightGradients[i][j][k] * weightGradients[i][j][k];
                
                // Compute bias-corrected first moment estimate
                double m_corrected = weightMoments1[i][j][k] / (1 - std::pow(beta1, timestep));
                
                // Compute bias-corrected second raw moment estimate
                double v_corrected = weightMoments2[i][j][k] / (1 - std::pow(beta2, timestep));
                
                // Update weights
                weights[i][j][k] -= learningRate * m_corrected / (std::sqrt(v_corrected) + epsilon);
            }
            
            // Update bias using Adam
            biasMoments1[i][j] = beta1 * biasMoments1[i][j] + (1 - beta1) * biasGradients[i][j];
            biasMoments2[i][j] = beta2 * biasMoments2[i][j] + 
                               (1 - beta2) * biasGradients[i][j] * biasGradients[i][j];
            
            double bias_m_corrected = biasMoments1[i][j] / (1 - std::pow(beta1, timestep));
            double bias_v_corrected = biasMoments2[i][j] / (1 - std::pow(beta2, timestep));
            
            biases[i][j] -= learningRate * bias_m_corrected / (std::sqrt(bias_v_corrected) + epsilon);
        }
    }
}

void NeuralNetwork::applyRegularization(std::vector<std::vector<std::vector<double>>>& weightGradients,
                                      double regularizationRate,
                                      RegularizationType regularizationType,
                                      double l1Ratio) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                switch (regularizationType) {
                    case RegularizationType::L1:
                        // L1 regularization: gradient += lambda * sign(weight)
                        weightGradients[i][j][k] += regularizationRate * ((weights#include "neural_network.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>

// Constructor with enhanced initialization
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                           ActivationFunction hiddenActivation,
                           ActivationFunction outputActivation,
                           InitializationMethod initMethod)
    : layerSizes(layerSizes),
      hiddenActivation(hiddenActivation),
      outputActivation(outputActivation) {
    
    // Use a random device to seed the random number generator
    std::random_device rd;
    rng = std::mt19937(rd());
    
    // Initialize weights using the specified method
    initializeWeights(initMethod);
    
    // Initialize optimizer state variables
    weightVelocities.resize(weights.size());
    biasVelocities.resize(biases.size());
    weightMoments1.resize(weights.size());
    biasMoments1.resize(biases.size());
    weightMoments2.resize(weights.size());
    biasMoments2.resize(biases.size());
    
    // Initialize batch normalization parameters
    batchNormLayers.resize(layerSizes.size() - 1);
    
    for (size_t i = 0; i < weights.size(); i++) {
        weightVelocities[i].resize(weights[i].size());
        weightMoments1[i].resize(weights[i].size());
        weightMoments2[i].resize(weights[i].size());
        
        for (size_t j = 0; j < weights[i].size(); j++) {
            weightVelocities[i][j].resize(weights[i][j].size(), 0.0);
            weightMoments1[i][j].resize(weights[i][j].size(), 0.0);
            weightMoments2[i][j].resize(weights[i][j].size(), 0.0);
        }
        
        biasVelocities[i].resize(biases[i].size(), 0.0);
        biasMoments1[i].resize(biases[i].size(), 0.0);
        biasMoments2[i].resize(biases[i].size(), 0.0);
        
        // Initialize batch normalization parameters for this layer
        batchNormLayers[i].gamma.resize(layerSizes[i + 1], 1.0);
        batchNormLayers[i].beta.resize(layerSizes[i + 1], 0.0);
        batchNormLayers[i].movingMean.resize(layerSizes[i + 1], 0.0);
        batchNormLayers[i].movingVar.resize(layerSizes[i + 1], 1.0);
    }
}

void NeuralNetwork::initializeWeights(InitializationMethod method) {
    weights.clear();
    biases.clear();
    
    // Create distributions for different initialization methods
    std::uniform_real_distribution<double> uniformDist(-0.5, 0.5);
    std::normal_distribution<double> normalDist(0.0, 1.0);
    
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        weights.push_back(std::vector<std::vector<double>>(
            layerSizes[i], std::vector<double>(layerSizes[i-1])));
        biases.push_back(std::vector<double>(layerSizes[i], 0.0));
        
        double scale = 1.0;
        
        switch (method) {
            case InitializationMethod::XAVIER:
                // Xavier/Glorot initialization
                scale = sqrt(6.0 / (layerSizes[i-1] + layerSizes[i]));
                break;
                
            case InitializationMethod::HE:
                // He initialization (better for ReLU)
                scale = sqrt(2.0 / layerSizes[i-1]);
                break;
                
            case InitializationMethod::LECUN:
                // LeCun initialization
                scale = sqrt(1.0 / layerSizes[i-1]);
                break;
                
            case InitializationMethod::NORMAL:
                // Normal distribution
                scale = 0.1;
                break;
                
            case InitializationMethod::UNIFORM:
                // Uniform distribution
                scale = 0.5;
                break;
        }
        
        // Initialize weights based on the selected method
        for (auto& neuron : weights.back()) {
            for (auto& weight : neuron) {
                if (method == InitializationMethod::NORMAL) {
                    weight = normalDist(rng) * scale;
                } else {
                    weight = uniformDist(rng) * scale;
                }
            }
        }
        
        // Initialize biases
        for (auto& bias : biases.back()) {
            if (method == InitializationMethod::NORMAL) {
                bias = normalDist(rng) * 0.1;
            } else {
                bias = 0.0; // Most modern networks initialize biases to zero
            }
        }
    }
}

std::vector<std::vector<double>> NeuralNetwork::forwardPass(const std::vector<double>& input) const {
    std::vector<std::vector<double>> activations;
    activations.push_back(input);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(layerSizes[i+1], 0.0);
        
        // Calculate weighted sum for each neuron
        for (size_t j = 0; j < weights[i].size(); ++j) {
            z[j] = biases[i][j];
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                z[j] += weights[i][j][k] * activations.back()[k];
            }
        }
        
        // Apply activation function
        ActivationFunction activation = (i == weights.size() - 1) ? outputActivation : hiddenActivation;
        activations.push_back(applyActivation(z, activation));
    }
    
    return activations;
}

std::vector<std::vector<double>> NeuralNetwork::forwardPassWithDropout(
    const std::vector<double>& input, 
    double dropoutRate,
    std::vector<std::vector<bool>>& dropoutMasks) {
    
    std::vector<std::vector<double>> activations;
    activations.push_back(input);
    
    dropoutMasks.clear();
    std::uniform_real_distribution<double> dropoutDist(0.0, 1.0);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(layerSizes[i+1], 0.0);
        
        // Calculate weighted sum for each neuron
        for (size_t j = 0; j < weights[i].size(); ++j) {
            z[j] = biases[i][j];
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                z[j] += weights[i][j][k] * activations.back()[k];
            }
        }
        
        // Apply activation function
        ActivationFunction activation = (i == weights.size() - 1) ? outputActivation : hiddenActivation;
        std::vector<double> layerActivation = applyActivation(z, activation);
        
        // Apply dropout (except for output layer)
        if (i < weights.size() - 1 && dropoutRate > 0) {
            std::vector<bool> dropoutMask(layerActivation.size(), false);
            
            for (size_t j = 0; j < layerActivation.size(); ++j) {
                // Keep neuron with probability (1-dropoutRate)
                if (dropoutDist(rng) >= dropoutRate) {
                    dropoutMask[j] = true;
                    // Scale by 1/(1-dropoutRate) to maintain expected value
                    layerActivation[j] /= (1.0 - dropoutRate);
                } else {
                    layerActivation[j] = 0.0;
                }
            }
            
            dropoutMasks.push_back(dropoutMask);
        }
        
        activations.push_back(layerActivation);
    }
    
    return activations;
}

std::vector<double> NeuralNetwork::applyActivation(const std::vector<double>& z, ActivationFunction activation) const {
    std::vector<double> result(z.size());
    
    switch (activation) {
        case ActivationFunction::SIGMOID:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = 1.0 / (1.0 + std::exp(-z[i]));
            }
            break;
            
        case ActivationFunction::TANH:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = std::tanh(z[i]);
            }
            break;
            
        case ActivationFunction::RELU:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = std::max(0.0, z[i]);
            }
            break;
            
        case ActivationFunction::LEAKY_RELU:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = z[i] > 0 ? z[i] : 0.01 * z[i];
            }
            break;
            
        case ActivationFunction::ELU:
            result = elu(z);
            break;
            
        case ActivationFunction::SWISH:
            result = swish(z);
            break;
            
        case ActivationFunction::MISH:
            result = mish(z);
            break;
            
        case ActivationFunction::SOFTMAX:
            result = softmax(z);
            break;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::applyActivationDerivative(const std::vector<double>& z, ActivationFunction activation) const {
    std::vector<double> result(z.size());
    std::vector<double> activated = applyActivation(z, activation);
    
    switch (activation) {
        case ActivationFunction::SIGMOID:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = activated[i] * (1.0 - activated[i]);
            }
            break;
            
        case ActivationFunction::TANH:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = 1.0 - activated[i] * activated[i];
            }
            break;
            
        case ActivationFunction::RELU:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = z[i] > 0 ? 1.0 : 0.0;
            }
            break;
            
        case ActivationFunction::LEAKY_RELU:
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = z[i] > 0 ? 1.0 : 0.01;
            }
            break;
            
        case ActivationFunction::ELU:
            for (size_t i = 0; i < z.size(); ++i) {
                if (z[i] > 0) {
                    result[i] = 1.0;
                } else {
                    // ELU derivative: alpha * exp(x) for x < 0
                    double alpha = 1.0;
                    result[i] = alpha * std::exp(z[i]);
                }
            }
            break;
            
        case ActivationFunction::SWISH:
            for (size_t i = 0; i < z.size(); ++i) {
                double sigmoid = 1.0 / (1.0 + std::exp(-z[i]));
                // Swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                result[i] = sigmoid + z[i] * sigmoid * (1.0 - sigmoid);
            }
            break;
            
        case ActivationFunction::MISH:
            for (size_t i = 0; i < z.size(); ++i) {
                // Mish derivative is complex; this is a simplified version
                double exp_x = std::exp(z[i]);
                double exp_2x = std::exp(2 * z[i]);
                double exp_3x = std::exp(3 * z[i]);
                double omega = 4.0 * (z[i] + 1.0) + 4.0 * exp_2x + exp_3x + exp_x * (4.0 * z[i] + 6.0);
                double delta = 2.0 * exp_x + exp_2x + 2.0;
                result[i] = exp_x * omega / (delta * delta);
            }
            break;
            
        case ActivationFunction::SOFTMAX:
            // For softmax, we typically handle the derivative differently in backpropagation
            // because we often use cross-entropy loss which simplifies the gradient
            for (size_t i = 0; i < z.size(); ++i) {
                // This is a placeholder; actual softmax derivative is a Jacobian matrix
                result[i] = activated[i] * (1.0 - activated[i]);
            }
            break;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    
    // Find max value for numerical stability
    double max_val = *std::max_element(z.begin(), z.end());
    
    // Compute exp(z_i - max_val) and sum
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        result[i] = std::exp(z[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    for (size_t i = 0; i < z.size(); ++i) {
        result[i] /= sum;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::elu(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    double alpha = 1.0; // Parameter controlling the value for negative inputs
    
    for (size_t i = 0; i < z.size(); ++i) {
        if (z[i] > 0) {
            result[i] = z[i];
        } else {
            result[i] = alpha * (std::exp(z[i]) - 1.0);
        }
    }
    
    return result;
}

std::vector<double> NeuralNetwork::swish(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    
    for (size_t i = 0; i < z.size(); ++i) {
        double sigmoid = 1.0 / (1.0 + std::exp(-z[i]));
        result[i] = z[i] * sigmoid;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::mish(const std::vector<double>& z) const {
    std::vector<double> result(z.size());
    
    for (size_t i = 0; i < z.size(); ++i) {
        // Softplus: log(1 + exp(x))
        double softplus = std::log1p(std::exp(z[i]));
        // Mish: x * tanh(softplus(x))
        result[i] = z[i] * std::tanh(softplus);
    }
    
    return result;
}

std::vector<double> NeuralNetwork::batchNormalize(const std::vector<double>& input, 
                                               size_t layerIndex,
                                               bool isTraining) {
    std::vector<double> result(input.size());
    const double epsilon = 1e-8;
    const double momentum = 0.9;
    
    if (isTraining) {
        // Calculate mean
        double mean = 0.0;
        for (size_t i = 0; i < input.size(); ++i) {
            mean += input[i];
        }
        mean /= input.size();
        
        // Calculate variance
        double variance = 0.0;
        for (size_t i = 0; i < input.size(); ++i) {
            double diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= input.size();
        
        // Update moving statistics
        for (size_t i = 0; i < input.size(); ++i) {
            batchNormLayers[layerIndex].movingMean[i] = 
                batchNormLayers[layerIndex].movingMean[i] * momentum + mean * (1 - momentum);
            batchNormLayers[layerIndex].movingVar[i] = 
                batchNormLayers[layerIndex].movingVar[i] * momentum + variance * (1 - momentum);
        }
        
        // Normalize
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = (input[i] - mean) / std::sqrt(variance + epsilon);
            // Scale and shift
            result[i] = result[i] * batchNormLayers[layerIndex].gamma[i] + batchNormLayers[layerIndex].beta[i];
        }
    } else {
        // Use stored mean and variance for inference
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = (input[i] - batchNormLayers[layerIndex].movingMean[i]) / 
                std::sqrt(batchNormLayers[layerIndex].movingVar[i] + epsilon);
            // Scale and shift
            result[i] = result[i] * batchNormLayers[layerIndex].gamma[i] + batchNormLayers[layerIndex].beta[i];
        }
    }
    
    return result;
}

void NeuralNetwork::applyAdamOptimizer(std::vector<std::vector<std::vector<double>>>& weightGradients,
                                     std::vector<std::vector<double>>& biasGradients,
                                     double learningRate, double beta1, double beta2, 
                                     double epsilon, int timestep) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                // Update biased first moment estimate
                weightMoments1[i][j][k] = beta1 * weightMoments1[i][j][k] + 
                                        (1 - beta1) * weightGradients[i][j][k];
                
                // Update biased second raw moment estimate
                weightMoments2[i][j][k] = beta2 * weightMoments2[i][j][k] + 
                                        (1 - beta2) * weightGradients[i][j][k] * weightGradients[i][j][k];
                
                // Compute bias-corrected first moment estimate
                double m_corrected = weightMoments1[i][j][k] / (1 - std::pow(beta1, timestep));
                
                // Compute bias-corrected second raw moment estimate
                double v_corrected = weightMoments2[i][j][k] / (1 - std::pow(beta2, timestep));
                
                // Update weights
                weights[i][j][k] -= learningRate * m_corrected / (std::sqrt(v_corrected) + epsilon);
            }
            
            // Update bias using Adam
            biasMoments1[i][j] = beta1 * biasMoments1[i][j] + (1 - beta1) * biasGradients[i][j];
            biasMoments2[i][j] = beta2 * biasMoments2[i][j] + 
                               (1 - beta2) * biasGradients[i][j] * biasGradients[i][j];
            
            double bias_m_corrected = biasMoments1[i][j] / (1 - std::pow(beta1, timestep));
            double bias_v_corrected = biasMoments2[i][j] / (1 - std::pow(beta2, timestep));
            
            biases[i][j] -= learningRate * bias_m_corrected / (std::sqrt(bias_v_corrected) + epsilon);
        }
    }
}

void NeuralNetwork::applyRegularization(std::vector<std::vector<std::vector<double>>>& weightGradients,
                                      double regularizationRate,
                                      RegularizationType regularizationType,
                                      double l1Ratio) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                switch (regularizationType) {
                    case RegularizationType::L1:
                    // L1 regularization: gradient += lambda * sign(weight)
                    weightGradients[i][j][k] += regularizationRate * (weights[i][j][k] > 0 ? 1.0 : -1.0);
                    break;
                    
                case RegularizationType::L2:
                    // L2 regularization: gradient += lambda * weight
                    weightGradients[i][j][k] += regularizationRate * weights[i][j][k];
                    break;
                    
                case RegularizationType::ELASTIC_NET:
                    // Elastic net: combination of L1 and L2
                    weightGradients[i][j][k] += regularizationRate * (
                        l1Ratio * (weights[i][j][k] > 0 ? 1.0 : -1.0) + 
                        (1.0 - l1Ratio) * weights[i][j][k]
                    );
                    break;
                    
                case RegularizationType::NONE:
                default:
                    break;
            }
        }
    }
}
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& outputs,
                    const TrainingOptions& options) {
// Use parallel training if enabled and we have enough data
if (options.useParallelProcessing && inputs.size() >= options.numThreads * 2) {
    trainInParallel(inputs, outputs, options);
    return;
}

// Initialize variables for early stopping
double bestValidationLoss = std::numeric_limits<double>::max();
int patience = options.earlyStoppingPatience;
bool shouldEarlyStop = false;

// Calculate number of batches
int batchCount = options.useBatchTraining ? 
    (inputs.size() + options.batchSize - 1) / options.batchSize : 1;

// Prepare for batch training
std::vector<int> indices(inputs.size());
for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<int>(i);
}

// Training loop
for (int epoch = 1; epoch <= options.epochs; ++epoch) {
    // Shuffle indices for stochastic/batch gradient descent
    if (options.useBatchTraining) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    // Calculate learning rate with decay
    double currentLearningRate = options.learningRate;
    if (options.useLearningRateDecay && epoch > 0) {
        currentLearningRate *= std::pow(options.learningRateDecayRate, 
                                      epoch / options.learningRateDecaySteps);
    }
    
    // Process each batch
    for (int batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        // Determine batch start and end indices
        int startIdx = batchIndex * options.batchSize;
        int endIdx = std::min(startIdx + options.batchSize, static_cast<int>(inputs.size()));
        
        // Initialize gradients
        std::vector<std::vector<std::vector<double>>> weightGradients(weights.size());
        std::vector<std::vector<double>> biasGradients(biases.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weightGradients[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weightGradients[i][j].resize(weights[i][j].size(), 0.0);
            }
            biasGradients[i].resize(biases[i].size(), 0.0);
        }
        
        // Process each example in the batch
        for (int idx = startIdx; idx < endIdx; ++idx) {
            int dataIndex = options.useBatchTraining ? indices[idx] : idx;
            
            // Forward pass with or without dropout
            std::vector<std::vector<double>> activations;
            std::vector<std::vector<bool>> dropoutMasks;
            
            if (options.useDropout) {
                activations = forwardPassWithDropout(inputs[dataIndex], options.dropoutRate, dropoutMasks);
            } else {
                activations = forwardPass(inputs[dataIndex]);
            }
            
            // Backpropagation
            std::vector<std::vector<std::vector<double>>> exampleGradients = 
                backpropagate(inputs[dataIndex], outputs[dataIndex], activations);
            
            // Update batch gradients
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    for (size_t k = 0; k < weights[i][j].size(); ++k) {
                        weightGradients[i][j][k] += exampleGradients[i][j][k] / (endIdx - startIdx);
                    }
                    
                    // Calculate bias gradient
                    double biasDelta = activations[i+1][j] - outputs[dataIndex][j];
                    biasGradients[i][j] += biasDelta / (endIdx - startIdx);
                }
            }
        }
        
        // Apply regularization
        if (options.regularizationType != RegularizationType::NONE && 
            options.regularizationRate > 0) {
            applyRegularization(weightGradients, options.regularizationRate, 
                              options.regularizationType, options.l1Ratio);
        }
        
        // Apply optimizer
        switch (options.optimizer) {
            case Optimizer::ADAM:
                applyAdamOptimizer(weightGradients, biasGradients, 
                                 currentLearningRate, options.beta1, 
                                 options.beta2, options.epsilon, epoch);
                break;
            
            case Optimizer::SGD:
            default:
                // Standard SGD with momentum
                for (size_t i = 0; i < weights.size(); ++i) {
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        for (size_t k = 0; k < weights[i][j].size(); ++k) {
                            // Apply momentum
                            weightVelocities[i][j][k] = options.momentum * weightVelocities[i][j][k] - 
                                                     currentLearningRate * weightGradients[i][j][k];
                            weights[i][j][k] += weightVelocities[i][j][k];
                        }
                        
                        // Update biases
                        biasVelocities[i][j] = options.momentum * biasVelocities[i][j] - 
                                            currentLearningRate * biasGradients[i][j];
                        biases[i][j] += biasVelocities[i][j];
                    }
                }
                break;
        }
    }
    
    // Early stopping check
    if (options.useEarlyStopping && epoch % 10 == 0) {
        double validationLoss = evaluateMSE(inputs, outputs);
        
        if (validationLoss < bestValidationLoss - options.earlyStoppingMinDelta) {
            bestValidationLoss = validationLoss;
            patience = options.earlyStoppingPatience;
        } else {
            patience--;
            if (patience <= 0) {
                std::cout << "Early stopping at epoch " << epoch << std::endl;
                break;
            }
        }
    }
}
}

void NeuralNetwork::trainInParallel(const std::vector<std::vector<double>>& inputs,
                              const std::vector<std::vector<double>>& outputs,
                              const TrainingOptions& options) {
// Initialize variables for early stopping
double bestValidationLoss = std::numeric_limits<double>::max();
int patience = options.earlyStoppingPatience;

// Calculate number of batches
int batchCount = options.useBatchTraining ? 
    (inputs.size() + options.batchSize - 1) / options.batchSize : 1;

// Prepare for batch training
std::vector<int> indices(inputs.size());
for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<int>(i);
}

// Training loop
for (int epoch = 1; epoch <= options.epochs; ++epoch) {
    // Shuffle indices for stochastic/batch gradient descent
    if (options.useBatchTraining) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    // Calculate learning rate with decay
    double currentLearningRate = options.learningRate;
    if (options.useLearningRateDecay && epoch > 0) {
        currentLearningRate *= std::pow(options.learningRateDecayRate, 
                                      epoch / options.learningRateDecaySteps);
    }
    
    // Process each batch
    for (int batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        // Determine batch start and end indices
        int startIdx = batchIndex * options.batchSize;
        int endIdx = std::min(startIdx + options.batchSize, static_cast<int>(inputs.size()));
        
        // Initialize gradients
        std::vector<std::vector<std::vector<double>>> weightGradients(weights.size());
        std::vector<std::vector<double>> biasGradients(biases.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weightGradients[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weightGradients[i][j].resize(weights[i][j].size(), 0.0);
            }
            biasGradients[i].resize(biases[i].size(), 0.0);
        }
        
        // Distribute work across threads
        int examplesPerThread = (endIdx - startIdx + options.numThreads - 1) / options.numThreads;
        std::vector<std::thread> threads;
        std::vector<std::vector<std::vector<std::vector<double>>>> threadWeightGradients(options.numThreads);
        std::vector<std::vector<std::vector<double>>> threadBiasGradients(options.numThreads);
        
        for (int t = 0; t < options.numThreads; ++t) {
            int threadStart = startIdx + t * examplesPerThread;
            int threadEnd = std::min(threadStart + examplesPerThread, endIdx);
            
            if (threadStart >= endIdx) {
                break;
            }
            
            threads.emplace_back([this, t, threadStart, threadEnd, &options, &indices, 
                                &inputs, &outputs, &threadWeightGradients, &threadBiasGradients]() {
                // Initialize thread-local gradients
                threadWeightGradients[t].resize(weights.size());
                threadBiasGradients[t].resize(biases.size());
                
                for (size_t i = 0; i < weights.size(); ++i) {
                    threadWeightGradients[t][i].resize(weights[i].size());
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        threadWeightGradients[t][i][j].resize(weights[i][j].size(), 0.0);
                    }
                    threadBiasGradients[t][i].resize(biases[i].size(), 0.0);
                }
                
                // Process examples assigned to this thread
                for (int idx = threadStart; idx < threadEnd; ++idx) {
                    int dataIndex = options.useBatchTraining ? indices[idx] : idx;
                    
                    // Forward pass with or without dropout
                    std::vector<std::vector<double>> activations;
                    std::vector<std::vector<bool>> dropoutMasks;
                    
                    if (options.useDropout) {
                        activations = forwardPassWithDropout(inputs[dataIndex], options.dropoutRate, dropoutMasks);
                    } else {
                        activations = forwardPass(inputs[dataIndex]);
                    }
                    
                    // Backpropagation
                    std::vector<std::vector<std::vector<double>>> exampleGradients = 
                        backpropagate(inputs[dataIndex], outputs[dataIndex], activations);
                    
                    // Update thread-local gradients
                    for (size_t i = 0; i < weights.size(); ++i) {
                        for (size_t j = 0; j < weights[i].size(); ++j) {
                            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                                threadWeightGradients[t][i][j][k] += 
                                    exampleGradients[i][j][k] / (threadEnd - threadStart);
                            }
                            
                            // Calculate bias gradient
                            double biasDelta = activations[i+1][j] - outputs[dataIndex][j];
                            threadBiasGradients[t][i][j] += biasDelta / (threadEnd - threadStart);
                        }
                    }
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Combine gradients from all threads
        for (int t = 0; t < options.numThreads; ++t) {
            if (t < threadWeightGradients.size()) {  // Check if thread produced any gradients
                for (size_t i = 0; i < weights.size(); ++i) {
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        for (size_t k = 0; k < weights[i][j].size(); ++k) {
                            weightGradients[i][j][k] += threadWeightGradients[t][i][j][k] / threads.size();
                        }
                        biasGradients[i][j] += threadBiasGradients[t][i][j] / threads.size();
                    }
                }
            }
        }
        
        // Apply regularization
        if (options.regularizationType != RegularizationType::NONE && 
            options.regularizationRate > 0) {
            applyRegularization(weightGradients, options.regularizationRate, 
                              options.regularizationType, options.l1Ratio);
        }
        
        // Apply optimizer
        switch (options.optimizer) {
            case Optimizer::ADAM:
                applyAdamOptimizer(weightGradients, biasGradients, 
                                 currentLearningRate, options.beta1, 
                                 options.beta2, options.epsilon, epoch);
                break;
            
            case Optimizer::SGD:
            default:
                // Standard SGD with momentum
                for (size_t i = 0; i < weights.size(); ++i) {
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        for (size_t k = 0; k < weights[i][j].size(); ++k) {
                            // Apply momentum
                            weightVelocities[i][j][k] = options.momentum * weightVelocities[i][j][k] - 
                                                     currentLearningRate * weightGradients[i][j][k];
                            weights[i][j][k] += weightVelocities[i][j][k];
                        }
                        
                        // Update biases
                        biasVelocities[i][j] = options.momentum * biasVelocities[i][j] - 
                                            currentLearningRate * biasGradients[i][j];
                        biases[i][j] += biasVelocities[i][j];
                    }
                }
                break;
        }
    }
    
    // Early stopping check
    if (options.useEarlyStopping && epoch % 10 == 0) {
        double validationLoss = evaluateMSE(inputs, outputs);
        
        if (validationLoss < bestValidationLoss - options.earlyStoppingMinDelta) {
            bestValidationLoss = validationLoss;
            patience = options.earlyStoppingPatience;
        } else {
            patience--;
            if (patience <= 0) {
                std::cout << "Early stopping at epoch " << epoch << std::endl;
                break;
            }
        }
    }
}
}

std::vector<std::vector<std::vector<double>>> NeuralNetwork::backpropagate(
const std::vector<double>& input,
const std::vector<double>& expectedOutput,
const std::vector<std::vector<double>>& activations) {

std::vector<std::vector<std::vector<double>>> weightGradients(weights.size());

// Initialize weight gradients
for (size_t i = 0; i < weights.size(); ++i) {
    weightGradients[i].resize(weights[i].size());
    for (size_t j = 0; j < weights[i].size(); ++j) {
        weightGradients[i][j].resize(weights[i][j].size(), 0.0);
    }
}

// Calculate output layer error
std::vector<double> delta = activations.back();
for (size_t i = 0; i < delta.size(); ++i) {
    delta[i] -= expectedOutput[i];
}

// Backpropagate error
for (int i = static_cast<int>(weights.size()) - 1; i >= 0; --i) {
    std::vector<double> nextDelta(layerSizes[i], 0.0);
    
    for (size_t j = 0; j < weights[i].size(); ++j) {
        for (size_t k = 0; k < weights[i][j].size(); ++k) {
            // Calculate weight gradient
            weightGradients[i][j][k] = delta[j] * activations[i][k];
            
            // Calculate delta for the previous layer
            nextDelta[k] += weights[i][j][k] * delta[j];
        }
    }
    
    if (i > 0) {
        // Apply activation derivative to the next delta
        ActivationFunction activation = (i == weights.size() - 1) ? outputActivation : hiddenActivation;
        std::vector<double> derivative = applyActivationDerivative(activations[i], activation);
        
        for (size_t j = 0; j < nextDelta.size(); ++j) {
            nextDelta[j] *= derivative[j];
        }
        
        delta = std::move(nextDelta);
    }
}

return weightGradients;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) const {
// Simple forward pass and return the output layer activations
std::vector<std::vector<double>> activations = forwardPass(input);
return activations.back();
}

double NeuralNetwork::evaluateAccuracy(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& expectedOutputs) const {
if (inputs.size() != expectedOutputs.size() || inputs.empty()) {
    return 0.0;
}

int correctPredictions = 0;

for (size_t i = 0; i < inputs.size(); ++i) {
    std::vector<double> prediction = predict(inputs[i]);
    
    // Find the index of the maximum value in prediction and expectedOutput
    size_t maxPredictionIdx = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
    size_t maxExpectedIdx = std::max_element(expectedOutputs[i].begin(), expectedOutputs[i].end()) - expectedOutputs[i].begin();
    
    if (maxPredictionIdx == maxExpectedIdx) {
        correctPredictions++;
    }
}

return static_cast<double>(correctPredictions) / inputs.size();
}

double NeuralNetwork::evaluateMSE(const std::vector<std::vector<double>>& inputs,
                            const std::vector<std::vector<double>>& expectedOutputs) const {
if (inputs.size() != expectedOutputs.size() || inputs.empty()) {
    return 0.0;
}

double totalError = 0.0;

for (size_t i = 0; i < inputs.size(); ++i) {
    std::vector<double> prediction = predict(inputs[i]);
    
    // Calculate squared error
    for (size_t j = 0; j < prediction.size(); ++j) {
        double error = prediction[j] - expectedOutputs[i][j];
        totalError += error * error;
    }
}

// Mean squared error
return totalError / (inputs.size() * expectedOutputs[0].size());
}

bool NeuralNetwork::saveModel(const std::string& filename) const {
std::ofstream file(filename, std::ios::binary);
if (!file.is_open()) {
    return false;
}

// Write network architecture
size_t numLayers = layerSizes.size();
file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));

for (size_t layer : layerSizes) {
    file.write(reinterpret_cast<const char*>(&layer), sizeof(layer));
}

// Write activation functions
file.write(reinterpret_cast<const char*>(&hiddenActivation), sizeof(hiddenActivation));
file.write(reinterpret_cast<const char*>(&outputActivation), sizeof(outputActivation));

// Write weights
for (const auto& layer : weights) {
    for (const auto& neuron : layer) {
        for (double weight : neuron) {
            file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
    }
}

// Write biases
for (const auto& layer : biases) {
    for (double bias : layer) {
        file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
    }
}

// Write batch normalization parameters
for (const auto& bnLayer : batchNormLayers) {
    for (double gamma : bnLayer.gamma) {
        file.write(reinterpret_cast<const char*>(&gamma), sizeof(gamma));
    }
    
    for (double beta : bnLayer.beta) {
        file.write(reinterpret_cast<const char*>(&beta), sizeof(beta));
    }
    
    for (double mean : bnLayer.movingMean) {
        file.write(reinterpret_cast<const char*>(&mean), sizeof(mean));
    }
    
    for (double var : bnLayer.movingVar) {
        file.write(reinterpret_cast<const char*>(&var), sizeof(var));
    }
}

file.close();
return true;
}

bool NeuralNetwork::loadModel(const std::string& filename) {
std::ifstream file(filename, std::ios::binary);
if (!file.is_open()) {
    return false;
}

// Read network architecture
size_t numLayers;
file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

layerSizes.resize(numLayers);
for (size_t& layer : layerSizes) {
    file.read(reinterpret_cast<char*>(&layer), sizeof(layer));
}

// Read activation functions
file.read(reinterpret_cast<char*>(&hiddenActivation), sizeof(hiddenActivation));
file.read(reinterpret_cast<char*>(&outputActivation), sizeof(outputActivation));

// Initialize weights and biases
weights.clear();
biases.clear();

for (size_t i = 1; i < layerSizes.size(); ++i) {
    weights.push_back(std::vector<std::vector<double>>(
        layerSizes[i], std::vector<double>(layerSizes[i-1])));
    biases.push_back(std::vector<double>(layerSizes[i]));
}

// Read weights
for (auto& layer : weights) {
    for (auto& neuron : layer) {
        for (double& weight : neuron) {
            file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        }
    }
}

// Read biases
for (auto& layer : biases) {
    for (double& bias : layer) {
        file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
    }
}

// Initialize optimizer state variables
weightVelocities.resize(weights.size());
biasVelocities.resize(biases.size());
weightMoments1.resize(weights.size());
biasMoments1.resize(biases.size());
weightMoments2.resize(weights.size());
biasMoments2.resize(biases.size());

// Initialize batch normalization parameters
batchNormLayers.resize(layerSizes.size() - 1);

for (size_t i = 0; i < weights.size(); i++) {
    weightVelocities[i].resize(weights[i].size());
    weightMoments1[i].resize(weights[i].size());
    weightMoments2[i].resize(weights[i].size());
    
    for (size_t j = 0; j < weights[i].size(); j++) {
        weightVelocities[i][j].resize(weights[i][j].size(), 0.0);
        weightMoments1[i][j].resize(weights[i][j].size(), 0.0);
        weightMoments2[i][j].resize(weights[i][j].size(), 0.0);
    }
    
    biasVelocities[i].resize(biases[i].size(), 0.0);
    biasMoments1[i].resize(biases[i].size(), 0.0);
    biasMoments2[i].resize(biases[i].size(), 0.0);
    
    // Initialize batch normalization parameters for this layer
    batchNormLayers[i].gamma.resize(layerSizes[i + 1], 1.0);
    batchNormLayers[i].beta.resize(layerSizes[i + 1], 0.0);
    batchNormLayers[i].movingMean.resize(layerSizes[i + 1], 0.0);
    batchNormLayers[i].movingVar.resize(layerSizes[i + 1], 1.0);
}

// Read batch normalization parameters
for (auto& bnLayer : batchNormLayers) {
    for (double& gamma : bnLayer.gamma) {
        file.read(reinterpret_cast<char*>(&gamma), sizeof(gamma));
    }
    
    for (double& beta : bnLayer.beta) {
        file.read(reinterpret_cast<char*>(&beta), sizeof(beta));
    }
    
    for (double& mean : bnLayer.movingMean) {
        file.read(reinterpret_cast<char*>(&mean), sizeof(mean));
    }
    
    for (double& var : bnLayer.movingVar) {
        file.read(reinterpret_cast<char*>(&var), sizeof(var));
    }
}

file.close();
return true;
}

std::string NeuralNetwork::getModelSummary() const {
std::stringstream ss;

ss << "Neural Network Summary:" << std::endl;
ss << "----------------------" << std::endl;

// Architecture
ss << "Architecture: ";
for (size_t i = 0; i < layerSizes.size(); ++i) {
    ss << layerSizes[i];
    if (i < layerSizes.size() - 1) {
        ss << " -> ";
    }
}
ss << std::endl;

// Activation functions
ss << "Hidden Activation: " << activationToString(hiddenActivation) << std::endl;
ss << "Output Activation: " << activationToString(outputActivation) << std::endl;

// Parameter count
int totalParams = 0;
int trainableParams = 0;

ss << "Layer Details:" << std::endl;
for (size_t i = 0; i < weights.size(); ++i) {
    int layerParams = 0;
    
    // Count weights
    for (const auto& neuron : weights[i]) {
        layerParams += neuron.size();
    }
    
    // Count biases
    layerParams += biases[i].size();
    
    // Count batch normalization parameters
    if (i < batchNormLayers.size()) {
        layerParams += 2 * batchNormLayers[i].gamma.size(); // gamma and beta
    }
    
    ss << "  Layer " << i+1 << ": " << layerSizes[i] << " -> " << layerSizes[i+1]
       << " (" << layerParams << " parameters)" << std::endl;
    
    totalParams += layerParams;
    trainableParams += layerParams;
}

ss << "----------------------" << std::endl;
ss << "Total Parameters: " << totalParams << std::endl;
ss << "Trainable Parameters: " << trainableParams << std::endl;

return ss.str();
}

std::string NeuralNetwork::activationToString(ActivationFunction activation) const {
switch (activation) {
    case ActivationFunction::SIGMOID: return "Sigmoid";
    case ActivationFunction::TANH: return "Tanh";
    case ActivationFunction::RELU: return "ReLU";
    case ActivationFunction::LEAKY_RELU: return "Leaky ReLU";
    case ActivationFunction::ELU: return "ELU";
    case ActivationFunction::SWISH: return "Swish";
    case ActivationFunction::MISH: return "Mish";
    case ActivationFunction::SOFT// Continuing from line 957
                case RegularizationType::L1:
                    // L1 regularization: gradient += lambda * sign(weight)
                    weightGradients[i][j][k] += regularizationRate * (weights[i][j][k] > 0 ? 1.0 : -1.0);
                    break;
                    
                case RegularizationType::L2:
                    // L2 regularization: gradient += lambda * weight
                    weightGradients[i][j][k] += regularizationRate * weights[i][j][k];
                    break;
                    
                case RegularizationType::ELASTIC_NET:
                    // Elastic net: combination of L1 and L2
                    weightGradients[i][j][k] += regularizationRate * (
                        l1Ratio * (weights[i][j][k] > 0 ? 1.0 : -1.0) + 
                        (1.0 - l1Ratio) * weights[i][j][k]
                    );
                    break;
                    
                case RegularizationType::NONE:
                default:
                    break;
            }
        }
    }
}
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& outputs,
                    const TrainingOptions& options) {
// Use parallel training if enabled and we have enough data
if (options.useParallelProcessing && inputs.size() >= options.numThreads * 2) {
    trainInParallel(inputs, outputs, options);
    return;
}

// Initialize variables for early stopping
double bestValidationLoss = std::numeric_limits<double>::max();
int patience = options.earlyStoppingPatience;
bool shouldEarlyStop = false;

// Calculate number of batches
int batchCount = options.useBatchTraining ? 
    (inputs.size() + options.batchSize - 1) / options.batchSize : 1;

// Prepare for batch training
std::vector<int> indices(inputs.size());
for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<int>(i);
}

// Training loop
for (int epoch = 1; epoch <= options.epochs; ++epoch) {
    // Shuffle indices for stochastic/batch gradient descent
    if (options.useBatchTraining) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    // Calculate learning rate with decay
    double currentLearningRate = options.learningRate;
    if (options.useLearningRateDecay && epoch > 0) {
        currentLearningRate *= std::pow(options.learningRateDecayRate, 
                                      epoch / options.learningRateDecaySteps);
    }
    
    // Process each batch
    for (int batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        // Determine batch start and end indices
        int startIdx = batchIndex * options.batchSize;
        int endIdx = std::min(startIdx + options.batchSize, static_cast<int>(inputs.size()));
        
        // Initialize gradients
        std::vector<std::vector<std::vector<double>>> weightGradients(weights.size());
        std::vector<std::vector<double>> biasGradients(biases.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weightGradients[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weightGradients[i][j].resize(weights[i][j].size(), 0.0);
            }
            biasGradients[i].resize(biases[i].size(), 0.0);
        }
        
        // Process each example in the batch
        for (int idx = startIdx; idx < endIdx; ++idx) {
            int dataIndex = options.useBatchTraining ? indices[idx] : idx;
            
            // Forward pass with or without dropout
            std::vector<std::vector<double>> activations;
            std::vector<std::vector<bool>> dropoutMasks;
            
            if (options.useDropout) {
                activations = forwardPassWithDropout(inputs[dataIndex], options.dropoutRate, dropoutMasks);
            } else {
                activations = forwardPass(inputs[dataIndex]);
            }
            
            // Backpropagation
            std::vector<std::vector<std::vector<double>>> exampleGradients = 
                backpropagate(inputs[dataIndex], outputs[dataIndex], activations);
            
            // Update batch gradients
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    for (size_t k = 0; k < weights[i][j].size(); ++k) {
                        weightGradients[i][j][k] += exampleGradients[i][j][k] / (endIdx - startIdx);
                    }
                    
                    // Calculate bias gradient
                    double biasDelta = activations[i+1][j] - outputs[dataIndex][j];
                    biasGradients[i][j] += biasDelta / (endIdx - startIdx);
                }
            }
        }
        
        // Apply regularization
        if (options.regularizationType != RegularizationType::NONE && 
            options.regularizationRate > 0) {
            applyRegularization(weightGradients, options.regularizationRate, 
                              options.regularizationType, options.l1Ratio);
        }
        
        // Apply optimizer
        switch (options.optimizer) {
            case Optimizer::ADAM:
                applyAdamOptimizer(weightGradients, biasGradients, 
                                 currentLearningRate, options.beta1, 
                                 options.beta2, options.epsilon, epoch);
                break;
            
            case Optimizer::SGD:
            default:
                // Standard SGD with momentum
                for (size_t i = 0; i < weights.size(); ++i) {
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        for (size_t k = 0; k < weights[i][j].size(); ++k) {
                            // Apply momentum
                            weightVelocities[i][j][k] = options.momentum * weightVelocities[i][j][k] - 
                                                     currentLearningRate * weightGradients[i][j][k];
                            weights[i][j][k] += weightVelocities[i][j][k];
                        }
                        
                        // Update biases
                        biasVelocities[i][j] = options.momentum * biasVelocities[i][j] - 
                                            currentLearningRate * biasGradients[i][j];
                        biases[i][j] += biasVelocities[i][j];
                    }
                }
                break;
        }
    }
    
    // Early stopping check
    if (options.useEarlyStopping && epoch % 10 == 0) {
        double validationLoss = evaluateMSE(inputs, outputs);
        
        if (validationLoss < bestValidationLoss - options.earlyStoppingMinDelta) {
            bestValidationLoss = validationLoss;
            patience = options.earlyStoppingPatience;
        } else {
            patience--;
            if (patience <= 0) {
                std::cout << "Early stopping at epoch " << epoch << std::endl;
                break;
            }
        }
    }
}
}

void NeuralNetwork::trainInParallel(const std::vector<std::vector<double>>& inputs,
                              const std::vector<std::vector<double>>& outputs,
                              const TrainingOptions& options) {
// Initialize variables for early stopping
double bestValidationLoss = std::numeric_limits<double>::max();
int patience = options.earlyStoppingPatience;

// Calculate number of batches
int batchCount = options.useBatchTraining ? 
    (inputs.size() + options.batchSize - 1) / options.batchSize : 1;

// Prepare for batch training
std::vector<int> indices(inputs.size());
for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<int>(i);
}

// Training loop
for (int epoch = 1; epoch <= options.epochs; ++epoch) {
    // Shuffle indices for stochastic/batch gradient descent
    if (options.useBatchTraining) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    
    // Calculate learning rate with decay
    double currentLearningRate = options.learningRate;
    if (options.useLearningRateDecay && epoch > 0) {
        currentLearningRate *= std::pow(options.learningRateDecayRate, 
                                      epoch / options.learningRateDecaySteps);
    }
    
    // Process each batch
    for (int batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        // Determine batch start and end indices
        int startIdx = batchIndex * options.batchSize;
        int endIdx = std::min(startIdx + options.batchSize, static_cast<int>(inputs.size()));
        
        // Initialize gradients
        std::vector<std::vector<std::vector<double>>> weightGradients(weights.size());
        std::vector<std::vector<double>> biasGradients(biases.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weightGradients[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weightGradients[i][j].resize(weights[i][j].size(), 0.0);
            }
            biasGradients[i].resize(biases[i].size(), 0.0);
        }
        
        // Distribute work across threads
        int examplesPerThread = (endIdx - startIdx + options.numThreads - 1) / options.numThreads;
        std::vector<std::thread> threads;
        std::vector<std::vector<std::vector<std::vector<double>>>> threadWeightGradients(options.numThreads);
        std::vector<std::vector<std::vector<double>>> threadBiasGradients(options.numThreads);
        
        for (int t = 0; t < options.numThreads; ++t) {
            int threadStart = startIdx + t * examplesPerThread;
            int threadEnd = std::min(threadStart + examplesPerThread, endIdx);
            
            if (threadStart >= endIdx) {
                break;
            }
            
            threads.emplace_back([this, t, threadStart, threadEnd, &options, &indices, 
                                &inputs, &outputs, &threadWeightGradients, &threadBiasGradients]() {
                // Initialize thread-local gradients
                threadWeightGradients[t].resize(weights.size());
                threadBiasGradients[t].resize(biases.size());
                
                for (size_t i = 0; i < weights.size(); ++i) {
                    threadWeightGradients[t][i].resize(weights[i].size());
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        threadWeightGradients[t][i][j].resize(weights[i][j].size(), 0.0);
                    }
                    threadBiasGradients[t][i].resize(biases[i].size(), 0.0);
                }
                
                // Process examples assigned to this thread
                for (int idx = threadStart; idx < threadEnd; ++idx) {
                    int dataIndex = options.useBatchTraining ? indices[idx] : idx;
                    
                    // Forward pass with or without dropout
                    std::vector<std::vector<double>> activations;
                    std::vector<std::vector<bool>> dropoutMasks;
                    
                    if (options.useDropout) {
                        activations = forwardPassWithDropout(inputs[dataIndex], options.dropoutRate, dropoutMasks);
                    } else {
                        activations = forwardPass(inputs[dataIndex]);
                    }
                    
                    // Backpropagation
                    std::vector<std::vector<std::vector<double>>> exampleGradients = 
                        backpropagate(inputs[dataIndex], outputs[dataIndex], activations);
                    
                    // Update thread-local gradients
                    for (size_t i = 0; i < weights.size(); ++i) {
                        for (size_t j = 0; j < weights[i].size(); ++j) {
                            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                                threadWeightGradients[t][i][j][k] += 
                                    exampleGradients[i][j][k] / (threadEnd - threadStart);
                            }
                            
                            // Calculate bias gradient
                            double biasDelta = activations[i+1][j] - outputs[dataIndex][j];
                            threadBiasGradients[t][i][j] += biasDelta / (threadEnd - threadStart);
                        }
                    }
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Combine gradients from all threads
        for (int t = 0; t < options.numThreads; ++t) {
            if (t < threadWeightGradients.size()) {  // Check if thread produced any gradients
                for (size_t i = 0; i < weights.size(); ++i) {
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        for (size_t k = 0; k < weights[i][j].size(); ++k) {
                            weightGradients[i][j][k] += threadWeightGradients[t][i][j][k] / threads.size();
                        }
                        biasGradients[i][j] += threadBiasGradients[t][i][j] / threads.size();
                    }
                }
            }
        }
        
        // Apply regularization
        if (options.regularizationType != RegularizationType::NONE && 
            options.regularizationRate > 0) {
            applyRegularization(weightGradients, options.regularizationRate, 
                              options.regularizationType, options.l1Ratio);
        }
        
        // Apply optimizer
        switch (options.optimizer) {
            case Optimizer::ADAM:
                applyAdamOptimizer(weightGradients, biasGradients, 
                                 currentLearningRate, options.beta1, 
                                 options.beta2, options.epsilon, epoch);
                break;
            
            case Optimizer::SGD:
            default:
                // Standard SGD with momentum
                for (size_t i = 0; i < weights.size(); ++i) {
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        for (size_t k = 0; k < weights[i][j].size(); ++k) {
                            // Apply momentum
                            weightVelocities[i][j][k] = options.momentum * weightVelocities[i][j][k] - 
                                                     currentLearningRate * weightGradients[i][j][k];
                            weights[i][j][k] += weightVelocities[i][j][k];
                        }
                        
                        // Update biases
                        biasVelocities[i][j] = options.momentum * biasVelocities[i][j] - 
                                            currentLearningRate * biasGradients[i][j];
                        biases[i][j] += biasVelocities[i][j];
                    }
                }
                break;
        }
    }
    
    // Early stopping check
    if (options.useEarlyStopping && epoch % 10 == 0) {
        double validationLoss = evaluateMSE(inputs, outputs);
        
        if (validationLoss < bestValidationLoss - options.earlyStoppingMinDelta) {
            bestValidationLoss = validationLoss;
            patience = options.earlyStoppingPatience;
        } else {
            patience--;
            if (patience <= 0) {
                std::cout << "Early stopping at epoch " << epoch << std::endl;
                break;
            }
        }
    }
}
}

std::vector<std::vector<std::vector<double>>> NeuralNetwork::backpropagate(
const std::vector<double>& input,
const std::vector<double>& expectedOutput,
const std::vector<std::vector<double>>& activations) {

std::vector<std::vector<std::vector<double>>> weightGradients(weights.size());

// Initialize weight gradients
for (size_t i = 0; i < weights.size(); ++i) {
    weightGradients[i].resize(weights[i].size());
    for (size_t j = 0; j < weights[i].size(); ++j) {
        weightGradients[i][j].resize(weights[i][j].size(), 0.0);
    }
}

// Calculate output layer error
std::vector<double> delta = activations.back();
for (size_t i = 0; i < delta.size(); ++i) {
    delta[i] -= expectedOutput[i];
}

// Backpropagate error
for (int i = static_cast<int>(weights.size()) - 1; i >= 0; --i) {
    std::vector<double> nextDelta(layerSizes[i], 0.0);
    
    for (size_t j = 0; j < weights[i].size(); ++j) {
        for (size_t k = 0; k < weights[i][j].size(); ++k) {
            // Calculate weight gradient
            weightGradients[i][j][k] = delta[j] * activations[i][k];
            
            // Calculate delta for the previous layer
            nextDelta[k] += weights[i][j][k] * delta[j];
        }
    }
    
    if (i > 0) {
        // Apply activation derivative to the next delta
        ActivationFunction activation = (i == weights.size() - 1) ? outputActivation : hiddenActivation;
        std::vector<double> derivative = applyActivationDerivative(activations[i], activation);
        
        for (size_t j = 0; j < nextDelta.size(); ++j) {
            nextDelta[j] *= derivative[j];
        }
        
        delta = std::move(nextDelta);
    }
}

return weightGradients;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) const {
// Simple forward pass and return the output layer activations
std::vector<std::vector<double>> activations = forwardPass(input);
return activations.back();
}

double NeuralNetwork::evaluateAccuracy(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& expectedOutputs) const {
if (inputs.size() != expectedOutputs.size() || inputs.empty()) {
    return 0.0;
}

int correctPredictions = 0;

for (size_t i = 0; i < inputs.size(); ++i) {
    std::vector<double> prediction = predict(inputs[i]);
    
    // Find the index of the maximum value in prediction and expectedOutput
    size_t maxPredictionIdx = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
    size_t maxExpectedIdx = std::max_element(expectedOutputs[i].begin(), expectedOutputs[i].end()) - expectedOutputs[i].begin();
    
    if (maxPredictionIdx == maxExpectedIdx) {
        correctPredictions++;
    }
}

return static_cast<double>(correctPredictions) / inputs.size();
}

double NeuralNetwork::evaluateMSE(const std::vector<std::vector<double>>& inputs,
                            const std::vector<std::vector<double>>& expectedOutputs) const {
if (inputs.size() != expectedOutputs.size() || inputs.empty()) {
    return 0.0;
}

double totalError = 0.0;

for (size_t i = 0; i < inputs.size(); ++i) {
    std::vector<double> prediction = predict(inputs[i]);
    
    // Calculate squared error
    for (size_t j = 0; j < prediction.size(); ++j) {
        double error = prediction[j] - expectedOutputs[i][j];
        totalError += error * error;
    }
}

// Mean squared error
return totalError / (inputs.size() * expectedOutputs[0].size());
}

bool NeuralNetwork::saveModel(const std::string& filename) const {
std::ofstream file(filename, std::ios::binary);
if (!file.is_open()) {
    return false;
}

// Write network architecture
size_t numLayers = layerSizes.size();
file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));

for (size_t layer : layerSizes) {
    file.write(reinterpret_cast<const char*>(&layer), sizeof(layer));
}

// Write activation functions
file.write(reinterpret_cast<const char*>(&hiddenActivation), sizeof(hiddenActivation));
file.write(reinterpret_cast<const char*>(&outputActivation), sizeof(outputActivation));

// Write weights
for (const auto& layer : weights) {
    for (const auto& neuron : layer) {
        for (double weight : neuron) {
            file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
    }
}

// Write biases
for (const auto& layer : biases) {
    for (double bias : layer) {
        file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
    }
}

// Write batch normalization parameters
for (const auto& bnLayer : batchNormLayers) {
    for (double gamma : bnLayer.gamma) {
        file.write(reinterpret_cast<const char*>(&gamma), sizeof(gamma));
    }
    
    for (double beta : bnLayer.beta) {
        file.write(reinterpret_cast<const char*>(&beta), sizeof(beta));
    }
    
    for (double mean : bnLayer.movingMean) {
        file.write(reinterpret_cast<const char*>(&mean), sizeof(mean));
    }
    
    for (double var : bnLayer.movingVar) {
        file.write(reinterpret_cast<const char*>(&var), sizeof(var));
    }
}

file.close();
return true;
}

bool NeuralNetwork::loadModel(const std::string& filename) {
std::ifstream file(filename, std::ios::binary);
if (!file.is_open()) {
    return false;
}

// Read network architecture
size_t numLayers;
file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

layerSizes.resize(numLayers);
for (size_t& layer : layerSizes) {
    file.read(reinterpret_cast<char*>(&layer), sizeof(layer));
}

// Read activation functions
file.read(reinterpret_cast<char*>(&hiddenActivation), sizeof(hiddenActivation));
file.read(reinterpret_cast<char*>(&outputActivation), sizeof(outputActivation));

// Initialize weights and biases
weights.clear();
biases.clear();

for (size_t i = 1; i < layerSizes.size(); ++i) {
    weights.push_back(std::vector<std::vector<double>>(
        layerSizes[i], std::vector<double>(layerSizes[i-1])));
    biases.push_back(std::vector<double>(layerSizes[i]));
}

// Read weights
for (auto& layer : weights) {
    for (auto& neuron : layer) {
        for (double& weight : neuron) {
            file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        }
    }
}

// Read biases
for (auto& layer : biases) {
    for (double& bias : layer) {
        file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
    }
}

// Initialize optimizer state variables
weightVelocities.resize(weights.size());
biasVelocities.resize(biases.size());
weightMoments1.resize(weights.size());
biasMoments1.resize(biases.size());
weightMoments2.resize(weights.size());
biasMoments2.resize(biases.size());

// Initialize batch normalization parameters
batchNormLayers.resize(layerSizes.size() - 1);

for (size_t i = 0; i < weights.size(); i++) {
    weightVelocities[i].resize(weights[i].size());
    weightMoments1[i].resize(weights[i].size());
    weightMoments2[i].resize(weights[i].size());
    
    for (size_t j = 0; j < weights[i].size(); j++) {
        weightVelocities[i][j].resize(weights[i][j].size(), 0.0);
        weightMoments1[i][j].resize(weights[i][j].size(), 0.0);
        weightMoments2[i][j].resize(weights[i][j].size(), 0.0);
    }
    
    biasVelocities[i].resize(biases[i].size(), 0.0);
    biasMoments1[i].resize(biases[i].size(), 0.0);
    biasMoments2[i].resize(biases[i].size(), 0.0);
    
    // Initialize batch normalization parameters for this layer
    batchNormLayers[i].gamma.resize(layerSizes[i + 1], 1.0);
    batchNormLayers[i].beta.resize(layerSizes[i + 1], 0.0);
    batchNormLayers[i].movingMean.resize(layerSizes[i + 1], 0.0);
    batchNormLayers[i].movingVar.resize(layerSizes[i + 1], 1.0);
}

// Read batch normalization parameters
for (auto& bnLayer : batchNormLayers) {
    for (double& gamma : bnLayer.gamma) {
        file.read(reinterpret_cast<char*>(&gamma), sizeof(gamma));
    }
    
    for (double& beta : bnLayer.beta) {
        file.read(reinterpret_cast<char*>(&beta), sizeof(beta));
    }
    
    for (double& mean : bnLayer.movingMean) {
        file.read(reinterpret_cast<char*>(&mean), sizeof(mean));
    }
    
    for (double& var : bnLayer.movingVar) {
        file.read(reinterpret_cast<char*>(&var), sizeof(var));
    }
}

file.close();
return true;
}

std::string NeuralNetwork::getModelSummary() const {
std::stringstream ss;

ss << "Neural Network Summary:" << std::endl;
ss << "----------------------" << std::endl;

// Architecture
ss << "Architecture: ";
for (size_t i = 0; i < layerSizes.size(); ++i) {
    ss << layerSizes[i];
    if (i < layerSizes.size() - 1) {
        ss << " -> ";
    }
}
ss << std::endl;

// Activation functions
ss << "Hidden Activation: " << activationToString(hiddenActivation) << std::endl;
ss << "Output Activation: " << activationToString(outputActivation) << std::endl;

// Parameter count
int totalParams = 0;
int trainableParams = 0;

ss << "Layer Details:" << std::endl;
for (size_t i = 0; i < weights.size(); ++i) {
    int layerParams = 0;
    
    // Count weights
    for (const auto& neuron : weights[i]) {
        layerParams += neuron.size();
    }
    
    // Count biases
    layerParams += biases[i].size();
    
    // Count batch normalization parameters
    if (i < batchNormLayers.size()) {
        layerParams += 2 * batchNormLayers[i].gamma.size(); // gamma and beta
    }
    
    ss << "  Layer " << i+1 << ": " << layerSizes[i] << " -> " << layerSizes[i+1]
       << " (" << layerParams << " parameters)" << std::endl;
    
    totalParams += layerParams;
    trainableParams += layerParams;
}

ss << "----------------------" << std::endl;
ss << "Total Parameters: " << totalParams << std::endl;
ss << "Trainable Parameters: " << trainableParams << std::endl;

return ss.str();
}

std::string NeuralNetwork::activationToString(ActivationFunction activation) const {
switch (activation) {
    case ActivationFunction::SIGMOID: return "Sigmoid";
    case ActivationFunction::TANH: return "Tanh";
    case ActivationFunction::RELU: return "ReLU";
    case ActivationFunction::LEAKY_RELU: return "Leaky ReLU";
    case ActivationFunction::ELU: return "ELU";
    case ActivationFunction::SWISH: return "Swish";
    case ActivationFunction::MISH: return "Mish";
    case ActivationFunction::SOFT
    case ActivationFunction::SOFTMAX: return "Softmax";
    default: return "Unknown";
}
}

// Implement advanced RMSProp optimizer
void NeuralNetwork::applyRMSPropOptimizer(
    std::vector<std::vector<std::vector<double>>>& weightGradients,
    std::vector<std::vector<double>>& biasGradients,
    double learningRate, double decayRate, double epsilon) {
    
    // Check if RMSProp cache is initialized
    if (weightRMSCache.empty()) {
        weightRMSCache.resize(weights.size());
        biasRMSCache.resize(biases.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weightRMSCache[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weightRMSCache[i][j].resize(weights[i][j].size(), 0.0);
            }
            biasRMSCache[i].resize(biases[i].size(), 0.0);
        }
    }
    
    // Apply RMSProp update rule
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                // Update moving average of squared gradients
                weightRMSCache[i][j][k] = decayRate * weightRMSCache[i][j][k] + 
                    (1 - decayRate) * weightGradients[i][j][k] * weightGradients[i][j][k];
                
                // Update weights using RMSProp formula
                weights[i][j][k] -= learningRate * weightGradients[i][j][k] / 
                    (std::sqrt(weightRMSCache[i][j][k]) + epsilon);
            }
            
            // Update bias using RMSProp
            biasRMSCache[i][j] = decayRate * biasRMSCache[i][j] + 
                (1 - decayRate) * biasGradients[i][j] * biasGradients[i][j];
            
            biases[i][j] -= learningRate * biasGradients[i][j] / 
                (std::sqrt(biasRMSCache[i][j]) + epsilon);
        }
    }
}

// Implement Adagrad optimizer
void NeuralNetwork::applyAdagradOptimizer(
    std::vector<std::vector<std::vector<double>>>& weightGradients,
    std::vector<std::vector<double>>& biasGradients,
    double learningRate, double epsilon) {
    
    // Check if Adagrad cache is initialized
    if (weightAdagradCache.empty()) {
        weightAdagradCache.resize(weights.size());
        biasAdagradCache.resize(biases.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weightAdagradCache[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weightAdagradCache[i][j].resize(weights[i][j].size(), 0.0);
            }
            biasAdagradCache[i].resize(biases[i].size(), 0.0);
        }
    }
    
    // Apply Adagrad update rule
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                // Update accumulated squared gradients
                weightAdagradCache[i][j][k] += weightGradients[i][j][k] * weightGradients[i][j][k];
                
                // Update weights using Adagrad formula
                weights[i][j][k] -= learningRate * weightGradients[i][j][k] / 
                    (std::sqrt(weightAdagradCache[i][j][k]) + epsilon);
            }
            
            // Update bias using Adagrad
            biasAdagradCache[i][j] += biasGradients[i][j] * biasGradients[i][j];
            
            biases[i][j] -= learningRate * biasGradients[i][j] / 
                (std::sqrt(biasAdagradCache[i][j]) + epsilon);
        }
    }
}

// Implement Nadam optimizer (Nesterov-accelerated Adaptive Moment Estimation)
void NeuralNetwork::applyNadamOptimizer(
    std::vector<std::vector<std::vector<double>>>& weightGradients,
    std::vector<std::vector<double>>& biasGradients,
    double learningRate, double beta1, double beta2, 
    double epsilon, int timestep) {
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                // Update biased first moment estimate
                weightMoments1[i][j][k] = beta1 * weightMoments1[i][j][k] + 
                                        (1 - beta1) * weightGradients[i][j][k];
                
                // Update biased second raw moment estimate
                weightMoments2[i][j][k] = beta2 * weightMoments2[i][j][k] + 
                                        (1 - beta2) * weightGradients[i][j][k] * weightGradients[i][j][k];
                
                // Compute bias-corrected first moment estimate
                double m_corrected = weightMoments1[i][j][k] / (1 - std::pow(beta1, timestep));
                
                // Compute bias-corrected second raw moment estimate
                double v_corrected = weightMoments2[i][j][k] / (1 - std::pow(beta2, timestep));
                
                // Compute the Nesterov momentum term
                double m_nesterov = (beta1 * m_corrected + (1 - beta1) * weightGradients[i][j][k]) / (1 - std::pow(beta1, timestep + 1));
                
                // Update weights using Nadam formula
                weights[i][j][k] -= learningRate * m_nesterov / (std::sqrt(v_corrected) + epsilon);
            }
            
            // Update bias using Nadam
            biasMoments1[i][j] = beta1 * biasMoments1[i][j] + (1 - beta1) * biasGradients[i][j];
            biasMoments2[i][j] = beta2 * biasMoments2[i][j] + 
                               (1 - beta2) * biasGradients[i][j] * biasGradients[i][j];
            
            double bias_m_corrected = biasMoments1[i][j] / (1 - std::pow(beta1, timestep));
            double bias_v_corrected = biasMoments2[i][j] / (1 - std::pow(beta2, timestep));
            double bias_m_nesterov = (beta1 * bias_m_corrected + (1 - beta1) * biasGradients[i][j]) / 
                                   (1 - std::pow(beta1, timestep + 1));
            
            biases[i][j] -= learningRate * bias_m_nesterov / (std::sqrt(bias_v_corrected) + epsilon);
        }
    }
}

// Implement AMSGrad optimizer (Adam variant with guaranteed convergence)
void NeuralNetwork::applyAMSGradOptimizer(
    std::vector<std::vector<std::vector<double>>>& weightGradients,
    std::vector<std::vector<double>>& biasGradients,
    double learningRate, double beta1, double beta2, 
    double epsilon, int timestep) {
    
    // Check if max past squared gradients caches are initialized
    if (weightMaxSquared.empty()) {
        weightMaxSquared.resize(weights.size());
        biasMaxSquared.resize(biases.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weightMaxSquared[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weightMaxSquared[i][j].resize(weights[i][j].size(), 0.0);
            }
            biasMaxSquared[i].resize(biases[i].size(), 0.0);
        }
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                // Update biased first moment estimate
                weightMoments1[i][j][k] = beta1 * weightMoments1[i][j][k] + 
                                        (1 - beta1) * weightGradients[i][j][k];
                
                // Update biased second raw moment estimate
                weightMoments2[i][j][k] = beta2 * weightMoments2[i][j][k] + 
                                        (1 - beta2) * weightGradients[i][j][k] * weightGradients[i][j][k];
                
                // Compute bias-corrected first moment estimate
                double m_corrected = weightMoments1[i][j][k] / (1 - std::pow(beta1, timestep));
                
                // Compute bias-corrected second raw moment estimate
                double v_corrected = weightMoments2[i][j][k] / (1 - std::pow(beta2, timestep));
                
                // Update maximum squared gradient (AMSGrad modification)
                weightMaxSquared[i][j][k] = std::max(weightMaxSquared[i][j][k], v_corrected);
                
                // Update weights using AMSGrad formula
                weights[i][j][k] -= learningRate * m_corrected / (std::sqrt(weightMaxSquared[i][j][k]) + epsilon);
            }
            
            // Update bias using AMSGrad
            biasMoments1[i][j] = beta1 * biasMoments1[i][j] + (1 - beta1) * biasGradients[i][j];
            biasMoments2[i][j] = beta2 * biasMoments2[i][j] + 
                               (1 - beta2) * biasGradients[i][j] * biasGradients[i][j];
            
            double bias_m_corrected = biasMoments1[i][j] / (1 - std::pow(beta1, timestep));
            double bias_v_corrected = biasMoments2[i][j] / (1 - std::pow(beta2, timestep));
            
            biasMaxSquared[i][j] = std::max(biasMaxSquared[i][j], bias_v_corrected);
            
            biases[i][j] -= learningRate * bias_m_corrected / (std::sqrt(biasMaxSquared[i][j]) + epsilon);
        }
    }
}

// Implement AdaMax optimizer
void NeuralNetwork::applyAdaMaxOptimizer(
    std::vector<std::vector<std::vector<double>>>& weightGradients,
    std::vector<std::vector<double>>& biasGradients,
    double learningRate, double beta1, double beta2, int timestep) {
    
    // Check if infinite norm caches are initialized
    if (weightInfNorm.empty()) {
        weightInfNorm.resize(weights.size());
        biasInfNorm.resize(biases.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            weightInfNorm[i].resize(weights[i].size());
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weightInfNorm[i][j].resize(weights[i][j].size(), 0.0);
            }
            biasInfNorm[i].resize(biases[i].size(), 0.0);
        }
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                // Update biased first moment estimate
                weightMoments1[i][j][k] = beta1 * weightMoments1[i][j][k] + 
                                        (1 - beta1) * weightGradients[i][j][k];
                
                // Update infinite norm (max norm) - AdaMax modification
                weightInfNorm[i][j][k] = std::max(beta2 * weightInfNorm[i][j][k], 
                                               std::abs(weightGradients[i][j][k]));
                
                // Compute bias-corrected first moment estimate
                double m_corrected = weightMoments1[i][j][k] / (1 - std::pow(beta1, timestep));
                
                // Update weights using AdaMax formula
                weights[i][j][k] -= learningRate * m_corrected / weightInfNorm[i][j][k];
            }
            
            // Update bias using AdaMax
            biasMoments1[i][j] = beta1 * biasMoments1[i][j] + (1 - beta1) * biasGradients[i][j];
            biasInfNorm[i][j] = std::max(beta2 * biasInfNorm[i][j], std::abs(biasGradients[i][j]));
            
            double bias_m_corrected = biasMoments1[i][j] / (1 - std::pow(beta1, timestep));
            
            biases[i][j] -= learningRate * bias_m_corrected / biasInfNorm[i][j];
        }
    }
}

// Additional helper function for GPU-like activation computation
std::vector<double> NeuralNetwork::computeActivationBatch(
    const std::vector<double>& inputs, ActivationFunction activation) {
    // This function simulates GPU-like parallel computation of activations
    // In a real implementation, this would be offloaded to GPU or use SIMD instructions
    
    std::vector<double> outputs(inputs.size());
    
    switch (activation) {
        case ActivationFunction::SIGMOID:
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
                outputs[i] = 1.0 / (1.0 + std::exp(-inputs[i]));
            }
            break;
            
        case ActivationFunction::RELU:
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
                outputs[i] = std::max(0.0, inputs[i]);
            }
            break;
            
        case ActivationFunction::TANH:
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
                outputs[i] = std::tanh(inputs[i]);
            }
            break;
            
        case ActivationFunction::LEAKY_RELU:
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
                outputs[i] = inputs[i] > 0 ? inputs[i] : 0.01 * inputs[i];
            }
            break;
            
        case ActivationFunction::ELU:
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
                outputs[i] = inputs[i] > 0 ? inputs[i] : (std::exp(inputs[i]) - 1.0);
            }
            break;
            
        case ActivationFunction::SWISH:
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
                double sigmoid = 1.0 / (1.0 + std::exp(-inputs[i]));
                outputs[i] = inputs[i] * sigmoid;
            }
            break;
            
        case ActivationFunction::MISH:
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
                double softplus = std::log1p(std::exp(inputs[i]));
                outputs[i] = inputs[i] * std::tanh(softplus);
            }
            break;
            
        case ActivationFunction::SOFTMAX:
            // Softmax needs special handling due to its dependence on all values
            outputs = softmax(inputs);
            break;
    }
    
    return outputs;
}

// Enhanced version of evaluateAccuracy that supports multi-class and binary classifications
double NeuralNetwork::evaluateClassificationAccuracy(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& expectedOutputs,
    double threshold) const {
    
    if (inputs.size() != expectedOutputs.size() || inputs.empty()) {
        return 0.0;
    }
    
    int correctPredictions = 0;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> prediction = predict(inputs[i]);
        
        // Handle multi-class classification (one-hot encoding)
        if (prediction.size() > 1) {
            size_t predictedClass = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
            size_t actualClass = std::max_element(expectedOutputs[i].begin(), expectedOutputs[i].end()) - expectedOutputs[i].begin();
            
            if (predictedClass == actualClass) {
                correctPredictions++;
            }
        }
        // Handle binary classification
        else if (prediction.size() == 1) {
            bool predictedPositive = prediction[0] >= threshold;
            bool actualPositive = expectedOutputs[i][0] >= 0.5; // Assuming 0.5 threshold for ground truth
            
            if (predictedPositive == actualPositive) {
                correctPredictions++;
            }
        }
    }
    
    return static_cast<double>(correctPredictions) / inputs.size();
}

// Calculate various evaluation metrics for binary classification
std::map<std::string, double> NeuralNetwork::evaluateBinaryClassificationMetrics(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& expectedOutputs,
    double threshold) const {
    
    if (inputs.size() != expectedOutputs.size() || inputs.empty()) {
        return {{"error", 1.0}};
    }
    
    int truePositives = 0, trueNegatives = 0;
    int falsePositives = 0, falseNegatives = 0;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> prediction = predict(inputs[i]);
        bool predictedPositive = prediction[0] >= threshold;
        bool actualPositive = expectedOutputs[i][0] >= 0.5;
        
        if (predictedPositive && actualPositive) {
            truePositives++;
        } else if (!predictedPositive && !actualPositive) {
            trueNegatives++;
        } else if (predictedPositive && !actualPositive) {
            falsePositives++;
        } else { // !predictedPositive && actualPositive
            falseNegatives++;
        }
    }
    
    double accuracy = static_cast<double>(truePositives + trueNegatives) / inputs.size();
    
    double precision = (truePositives + falsePositives > 0) ? 
        static_cast<double>(truePositives) / (truePositives + falsePositives) : 0.0;
    
    double recall = (truePositives + falseNegatives > 0) ?
        static_cast<double>(truePositives) / (truePositives + falseNegatives) : 0.0;
    
    double f1Score = (precision + recall > 0) ?
        2 * precision * recall / (precision + recall) : 0.0;
    
    double specificity = (trueNegatives + falsePositives > 0) ?
        static_cast<double>(trueNegatives) / (trueNegatives + falsePositives) : 0.0;
    
    return {
        {"accuracy", accuracy},
        {"precision", precision},
        {"recall", recall},
        {"f1_score", f1Score},
        {"specificity", specificity},
        {"true_positives", static_cast<double>(truePositives)},
        {"true_negatives", static_cast<double>(trueNegatives)},
        {"false_positives", static_cast<double>(falsePositives)},
        {"false_negatives", static_cast<double>(falseNegatives)}
    };
}

// Generate a confusion matrix for multi-class classification
std::vector<std::vector<int>> NeuralNetwork::generateConfusionMatrix(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& expectedOutputs) const {
    
    if (inputs.empty() || expectedOutputs.empty()) {
        return {};
    }
    
    // Determine number of classes
    size_t numClasses = expectedOutputs[0].size();
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> prediction = predict(inputs[i]);
        size_t predictedClass = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
        size_t actualClass = std::max_element(expectedOutputs[i].begin(), expectedOutputs[i].end()) - expectedOutputs[i].begin();
        
        if (predictedClass < numClasses && actualClass < numClasses) {
            confusionMatrix[actualClass][predictedClass]++;
        }
    }
    
    return confusionMatrix;
}