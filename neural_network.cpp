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
#include <thread>

// Define InitializationMethod enum if not already in neural_network.h
enum class InitializationMethod {
    XAVIER,  // Good for tanh
    HE,      // Good for ReLU
    LECUN,   // Another scaling approach
    NORMAL,  // Normal distribution
    UNIFORM  // Uniform distribution
};

// Constructor with enhanced initialization
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                           ActivationFunction hiddenActivation,
                           ActivationFunction outputActivation)
    : layerSizes(layerSizes),
      hiddenActivation(hiddenActivation),
      outputActivation(outputActivation) {
    
    // Use a random device to seed the random number generator
    std::random_device rd;
    rng = std::mt19937(rd());
    
    // Initialize weights using He initialization (good for ReLU)
    initializeWeights(InitializationMethod::HE);
    
    // Initialize optimizer state variables
    velocities.resize(layerSizes.size() - 1);
    cacheWeights.resize(layerSizes.size() - 1);
    velocitiesBias.resize(layerSizes.size() - 1);
    cacheBias.resize(layerSizes.size() - 1);
    
    for (size_t i = 0; i < weights.size(); i++) {
        velocities[i].resize(weights[i].size());
        cacheWeights[i].resize(weights[i].size());
        
        for (size_t j = 0; j < weights[i].size(); j++) {
            velocities[i][j].resize(weights[i][j].size(), 0.0);
            cacheWeights[i][j].resize(weights[i][j].size(), 0.0);
        }
        
        velocitiesBias[i].resize(biases[i].size(), 0.0);
        cacheBias[i].resize(biases[i].size(), 0.0);
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

std::function<double(double)> NeuralNetwork::getActivationFunction(ActivationFunction func) {
    switch (func) {
        case ActivationFunction::SIGMOID:
            return [](double x) { return 1.0 / (1.0 + exp(-x)); };
        case ActivationFunction::RELU:
            return [](double x) { return x > 0 ? x : 0; };
        case ActivationFunction::LEAKY_RELU:
            return [](double x) { return x > 0 ? x : 0.01 * x; };
        case ActivationFunction::TANH:
            return [](double x) { return tanh(x); };
        case ActivationFunction::LINEAR:
        default:
            return [](double x) { return x; };
    }
}

std::function<double(double)> NeuralNetwork::getActivationDerivative(ActivationFunction func) {
    switch (func) {
        case ActivationFunction::SIGMOID:
            return [](double x) { 
                double s = 1.0 / (1.0 + exp(-x));
                return s * (1.0 - s); 
            };
        case ActivationFunction::RELU:
            return [](double x) { return x > 0 ? 1.0 : 0.0; };
        case ActivationFunction::LEAKY_RELU:
            return [](double x) { return x > 0 ? 1.0 : 0.01; };
        case ActivationFunction::TANH:
            return [](double x) { 
                double t = tanh(x);
                return 1.0 - t * t; 
            };
        case ActivationFunction::LINEAR:
        default:
            return [](double x) { return 1.0; };
    }
}

std::vector<double> NeuralNetwork::applySoftmax(const std::vector<double>& inputs) {
    std::vector<double> result(inputs.size());
    
    // Find max value for numerical stability
    double max_val = *std::max_element(inputs.begin(), inputs.end());
    
    // Compute exp(x_i - max_val) and sum
    double sum = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        result[i] = std::exp(inputs[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] /= sum;
    }
    
    return result;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) {
    std::vector<double> current = inputs;
    
    // Forward pass through each layer
    for (size_t i = 0; i < layerSizes.size() - 1; i++) {
        std::vector<double> next(layerSizes[i + 1], 0.0);
        
        // For each neuron in the next layer
        for (size_t j = 0; j < next.size(); j++) {
            // Add bias
            next[j] = biases[i][j];
            
            // Compute weighted sum of inputs
            for (size_t k = 0; k < current.size(); k++) {
                next[j] += weights[i][j][k] * current[k];
            }
        }
        
        // Apply activation function
        ActivationFunction activation = (i == layerSizes.size() - 2) ? 
                                        outputActivation : hiddenActivation;
        
        if (activation == ActivationFunction::SOFTMAX) {
            current = applySoftmax(next);
        } else {
            auto activationFunc = getActivationFunction(activation);
            for (size_t j = 0; j < next.size(); j++) {
                current[j] = activationFunc(next[j]);
            }
        }
    }
    
    return current;
}

double NeuralNetwork::calculateLoss(const std::vector<double>& outputs, 
                                    const std::vector<double>& targets) {
    // Mean Squared Error loss
    double sum = 0.0;
    for (size_t i = 0; i < outputs.size(); i++) {
        double diff = outputs[i] - targets[i];
        sum += diff * diff;
    }
    return sum / outputs.size();
}

double NeuralNetwork::evaluateAccuracy(const std::vector<std::vector<double>>& inputs, 
                                     const std::vector<std::vector<double>>& targets) {
    if (inputs.size() != targets.size() || inputs.empty()) {
        return 0.0;
    }
    
    int correctPredictions = 0;
    
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> output = predict(inputs[i]);
        
        // Find index of max value in both output and target
        auto maxElementOutput = std::max_element(output.begin(), output.end());
        auto maxElementTarget = std::max_element(targets[i].begin(), targets[i].end());
        
        if (std::distance(output.begin(), maxElementOutput) == 
            std::distance(targets[i].begin(), maxElementTarget)) {
            correctPredictions++;
        }
    }
    
    return static_cast<double>(correctPredictions) / inputs.size();
}

double NeuralNetwork::evaluateLoss(const std::vector<std::vector<double>>& inputs, 
                                 const std::vector<std::vector<double>>& targets) {
    if (inputs.size() != targets.size() || inputs.empty()) {
        return 0.0;
    }
    
    double totalLoss = 0.0;
    
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> output = predict(inputs[i]);
        totalLoss += calculateLoss(output, targets[i]);
    }
    
    return totalLoss / inputs.size();
}

// The rest of your functions can be added here...
// Make sure to check for and remove any duplicated code sections
// The previous implementations are good, we just need to fix the duplicate section and
// clean up the remainder of the file (from the void NeuralNetwork::train method onwards)

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, 
    const std::vector<std::vector<double>>& targets, 
    const TrainingOptions& options) {
if (inputs.size() != targets.size() || inputs.empty()) {
std::cerr << "Error: Mismatched input/target data or empty dataset." << std::endl;
return;
}

// Initialize optimizer state variables if needed
initializeOptimizer(options);

// For early stopping
double bestLoss = std::numeric_limits<double>::max();
int patience = options.patienceEpochs;

// For mini-batch processing
std::vector<int> indices(inputs.size());
for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
indices[i] = i;
}

// Training loop
for (int epoch = 0; epoch < options.epochs; epoch++) {
// Shuffle data for stochastic gradient descent
std::shuffle(indices.begin(), indices.end(), rng);

double epochLoss = 0.0;

// Process each mini-batch
for (int batch = 0; batch < static_cast<int>(inputs.size()); batch += options.batchSize) {
int batchEnd = std::min(batch + options.batchSize, static_cast<int>(inputs.size()));

// Create batch data
std::vector<std::vector<double>> batchInputs;
std::vector<std::vector<double>> batchTargets;

for (int idx = batch; idx < batchEnd; idx++) {
batchInputs.push_back(inputs[indices[idx]]);
batchTargets.push_back(targets[indices[idx]]);
}

// Train on this batch
double batchLoss = trainOnBatch(batchInputs, batchTargets, options);
epochLoss += batchLoss * (batchEnd - batch);
}

// Calculate average loss for the epoch
epochLoss /= inputs.size();

// Early stopping check
if (options.useEarlyStopping) {
if (epochLoss < bestLoss - options.earlyStoppingDelta) {
bestLoss = epochLoss;
patience = options.patienceEpochs;
} else {
patience--;
if (patience <= 0) {
std::cout << "Early stopping at epoch " << epoch << std::endl;
break;
}
}
}

// Print progress every few epochs
if ((epoch + 1) % 10 == 0 || epoch == 0) {
std::cout << "Epoch " << (epoch + 1) << "/" << options.epochs 
<< ", Loss: " << epochLoss << std::endl;
}
}
}

void NeuralNetwork::initializeOptimizer(const TrainingOptions& options) {
// This function ensures all optimizer state variables are properly initialized
// We've already initialized the base variables in the constructor

// For more complex optimizers like Adam, initialize additional state if needed
if (options.optimizer == Optimizer::ADAM) {
// Already initialized in constructor
}
}

void NeuralNetwork::updateWeightsWithOptimizer(int layerIndex, int neuronIndex, int weightIndex, 
    double gradient, const TrainingOptions& options, int t) {
switch (options.optimizer) {
case Optimizer::MOMENTUM:
{
// Momentum update
velocities[layerIndex][neuronIndex][weightIndex] = 
options.momentum * velocities[layerIndex][neuronIndex][weightIndex] - 
options.learningRate * gradient;
weights[layerIndex][neuronIndex][weightIndex] += 
velocities[layerIndex][neuronIndex][weightIndex];
break;
}

case Optimizer::ADAM:
{
// Adam optimizer update - Note the curly braces to create a block
velocities[layerIndex][neuronIndex][weightIndex] = 
options.beta1 * velocities[layerIndex][neuronIndex][weightIndex] + 
(1 - options.beta1) * gradient;

cacheWeights[layerIndex][neuronIndex][weightIndex] = 
options.beta2 * cacheWeights[layerIndex][neuronIndex][weightIndex] + 
(1 - options.beta2) * gradient * gradient;

// Bias correction
double m_corrected = velocities[layerIndex][neuronIndex][weightIndex] / 
(1 - std::pow(options.beta1, t));
double v_corrected = cacheWeights[layerIndex][neuronIndex][weightIndex] / 
(1 - std::pow(options.beta2, t));

weights[layerIndex][neuronIndex][weightIndex] -= 
options.learningRate * m_corrected / (std::sqrt(v_corrected) + options.epsilon);
break;
}

case Optimizer::RMSPROP:
{
// RMSProp update
cacheWeights[layerIndex][neuronIndex][weightIndex] = 
0.9 * cacheWeights[layerIndex][neuronIndex][weightIndex] + 
0.1 * gradient * gradient;

weights[layerIndex][neuronIndex][weightIndex] -= 
options.learningRate * gradient / 
(std::sqrt(cacheWeights[layerIndex][neuronIndex][weightIndex]) + options.epsilon);
break;
}

case Optimizer::SGD:
default:
{
// Standard SGD update
weights[layerIndex][neuronIndex][weightIndex] -= options.learningRate * gradient;
break;
}
}
}

void NeuralNetwork::updateBiasWithOptimizer(int layerIndex, int neuronIndex, 
double gradient, const TrainingOptions& options, int t) {
switch (options.optimizer) {
case Optimizer::MOMENTUM:
{
// Momentum update for bias
velocitiesBias[layerIndex][neuronIndex] = 
options.momentum * velocitiesBias[layerIndex][neuronIndex] - 
options.learningRate * gradient;
biases[layerIndex][neuronIndex] += velocitiesBias[layerIndex][neuronIndex];
break;
}

case Optimizer::ADAM:
{
// Adam optimizer update for bias
velocitiesBias[layerIndex][neuronIndex] = 
options.beta1 * velocitiesBias[layerIndex][neuronIndex] + 
(1 - options.beta1) * gradient;

cacheBias[layerIndex][neuronIndex] = 
options.beta2 * cacheBias[layerIndex][neuronIndex] + 
(1 - options.beta2) * gradient * gradient;

// Bias correction
double m_corrected = velocitiesBias[layerIndex][neuronIndex] / 
(1 - std::pow(options.beta1, t));
double v_corrected = cacheBias[layerIndex][neuronIndex] / 
(1 - std::pow(options.beta2, t));

biases[layerIndex][neuronIndex] -= 
options.learningRate * m_corrected / (std::sqrt(v_corrected) + options.epsilon);
break;
}

case Optimizer::RMSPROP:
{
// RMSProp update for bias
cacheBias[layerIndex][neuronIndex] = 
0.9 * cacheBias[layerIndex][neuronIndex] + 
0.1 * gradient * gradient;

biases[layerIndex][neuronIndex] -= 
options.learningRate * gradient / 
(std::sqrt(cacheBias[layerIndex][neuronIndex]) + options.epsilon);
break;
}

case Optimizer::SGD:
default:
{
    // Standard SGD update for bias
    biases[layerIndex][neuronIndex] -= options.learningRate * gradient;
break;
}
}
}

double NeuralNetwork::trainOnBatch(const std::vector<std::vector<double>>& batchInputs,
            const std::vector<std::vector<double>>& batchTargets,
            const TrainingOptions& options) {
if (batchInputs.size() != batchTargets.size() || batchInputs.empty()) {
std::cerr << "Error: Mismatched input/target data or empty batch." << std::endl;
return 0.0;
}

double batchLoss = 0.0;

// Reset gradients for this batch
std::vector<std::vector<std::vector<double>>> weightGradients(weights.size());
std::vector<std::vector<double>> biasGradients(biases.size());

for (size_t i = 0; i < weights.size(); i++) {
weightGradients[i].resize(weights[i].size());
for (size_t j = 0; j < weights[i].size(); j++) {
weightGradients[i][j].resize(weights[i][j].size(), 0.0);
}
biasGradients[i].resize(biases[i].size(), 0.0);
}

// Process each example in the batch
for (size_t dataIndex = 0; dataIndex < batchInputs.size(); dataIndex++) {
// Forward pass
std::vector<std::vector<double>> layerOutputs;
layerOutputs.push_back(batchInputs[dataIndex]);

for (size_t layer = 0; layer < layerSizes.size() - 1; layer++) {
std::vector<double> layerInput = layerOutputs.back();
std::vector<double> layerOutput(layerSizes[layer + 1], 0.0);

// For each neuron in this layer
for (size_t j = 0; j < layerOutput.size(); j++) {
// Add bias
layerOutput[j] = biases[layer][j];

// Compute weighted sum
for (size_t k = 0; k < layerInput.size(); k++) {
layerOutput[j] += weights[layer][j][k] * layerInput[k];
}
}

// Apply activation function
ActivationFunction activation = (layer == layerSizes.size() - 2) ? 
                     outputActivation : hiddenActivation;

if (activation == ActivationFunction::SOFTMAX) {
layerOutputs.push_back(applySoftmax(layerOutput));
} else {
auto activationFunc = getActivationFunction(activation);
for (size_t j = 0; j < layerOutput.size(); j++) {
layerOutput[j] = activationFunc(layerOutput[j]);
}
layerOutputs.push_back(layerOutput);
}
}

// Compute the loss
double loss = calculateLoss(layerOutputs.back(), batchTargets[dataIndex]);
batchLoss += loss;

// Backpropagation
std::vector<double> errors = layerOutputs.back();
for (size_t j = 0; j < errors.size(); j++) {
errors[j] = errors[j] - batchTargets[dataIndex][j];
}

// Backpropagate the error through each layer
for (int layer = static_cast<int>(layerSizes.size()) - 2; layer >= 0; layer--) {
// For output layer, compute directly, otherwise backpropagate error
if (layer < static_cast<int>(layerSizes.size()) - 2) {
std::vector<double> nextLayerErrors(layerSizes[layer + 1], 0.0);

for (size_t j = 0; j < layerSizes[layer + 1]; j++) {
for (size_t k = 0; k < layerSizes[layer + 2]; k++) {
  nextLayerErrors[j] += errors[k] * weights[layer + 1][k][j];
}

// Apply derivative of activation function
ActivationFunction activation = hiddenActivation;
auto derivativeFunc = getActivationDerivative(activation);
nextLayerErrors[j] *= derivativeFunc(layerOutputs[layer + 1][j]);
}

errors = nextLayerErrors;
}

// Update gradients for this layer
for (size_t j = 0; j < layerSizes[layer + 1]; j++) {
for (size_t k = 0; k < layerSizes[layer]; k++) {
weightGradients[layer][j][k] += errors[j] * layerOutputs[layer][k];
}
biasGradients[layer][j] += errors[j];
}
}
}

// Apply regularization if needed
if (options.useL1Regularization) {
for (size_t i = 0; i < weights.size(); i++) {
for (size_t j = 0; j < weights[i].size(); j++) {
for (size_t k = 0; k < weights[i][j].size(); k++) {
weightGradients[i][j][k] += options.l1Lambda * (weights[i][j][k] > 0 ? 1 : -1);
}
}
}
}

if (options.useL2Regularization) {
for (size_t i = 0; i < weights.size(); i++) {
for (size_t j = 0; j < weights[i].size(); j++) {
for (size_t k = 0; k < weights[i][j].size(); k++) {
weightGradients[i][j][k] += options.l2Lambda * weights[i][j][k];
}
}
}
}

// Apply gradients using the chosen optimizer
for (size_t i = 0; i < weights.size(); i++) {
for (size_t j = 0; j < weights[i].size(); j++) {
for (size_t k = 0; k < weights[i][j].size(); k++) {
// Average the gradient over the batch
double gradient = weightGradients[i][j][k] / batchInputs.size();
updateWeightsWithOptimizer(i, j, k, gradient, options, 1);
}

// Update bias
double biasGradient = biasGradients[i][j] / batchInputs.size();
updateBiasWithOptimizer(i, j, biasGradient, options, 1);
}
}

return batchLoss / batchInputs.size();
}

std::map<std::string, double> NeuralNetwork::evaluateBinaryClassificationMetrics(
const std::vector<std::vector<double>>& inputs, 
const std::vector<std::vector<double>>& targets) {

if (inputs.size() != targets.size() || inputs.empty()) {
return {
{"accuracy", 0.0},
{"precision", 0.0},
{"recall", 0.0},
{"f1_score", 0.0}
};
}

int truePositives = 0;
int falsePositives = 0;
int trueNegatives = 0;
int falseNegatives = 0;

for (size_t i = 0; i < inputs.size(); i++) {
std::vector<double> output = predict(inputs[i]);

// For binary classification, we threshold at 0.5
bool predicted = output[0] >= 0.5;
bool actual = targets[i][0] >= 0.5;

if (predicted && actual) {
truePositives++;
} else if (predicted && !actual) {
falsePositives++;
} else if (!predicted && !actual) {
trueNegatives++;
} else if (!predicted && actual) {
falseNegatives++;
}
}

double accuracy = static_cast<double>(truePositives + trueNegatives) / 
static_cast<double>(inputs.size());

double precision = (truePositives + falsePositives > 0) ? 
static_cast<double>(truePositives) / 
static_cast<double>(truePositives + falsePositives) : 0.0;

double recall = (truePositives + falseNegatives > 0) ? 
static_cast<double>(truePositives) / 
static_cast<double>(truePositives + falseNegatives) : 0.0;

double f1Score = (precision + recall > 0) ? 
2.0 * (precision * recall) / (precision + recall) : 0.0;

return {
{"accuracy", accuracy},
{"precision", precision},
{"recall", recall},
{"f1_score", f1Score}
};
}

void NeuralNetwork::saveModel(const std::string& filename) {
std::ofstream file(filename);
if (!file.is_open()) {
std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
return;
}

// Write network architecture
file << layerSizes.size() << "\n";
for (int size : layerSizes) {
file << size << " ";
}
file << "\n";

// Write activation functions (as integers)
file << static_cast<int>(hiddenActivation) << " " 
<< static_cast<int>(outputActivation) << "\n";

// Write weights
for (size_t i = 0; i < weights.size(); i++) {
for (size_t j = 0; j < weights[i].size(); j++) {
for (size_t k = 0; k < weights[i][j].size(); k++) {
file << weights[i][j][k] << " ";
}
file << "\n";
}
}

// Write biases
for (size_t i = 0; i < biases.size(); i++) {
for (size_t j = 0; j < biases[i].size(); j++) {
file << biases[i][j] << " ";
}
file << "\n";
}

file.close();
}

void NeuralNetwork::loadModel(const std::string& filename) {
std::ifstream file(filename);
if (!file.is_open()) {
std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
return;
}

// Read network architecture
int numLayers;
file >> numLayers;

std::vector<int> newLayerSizes(numLayers);
for (int i = 0; i < numLayers; i++) {
file >> newLayerSizes[i];
}

// Read activation functions
int hiddenActivationInt, outputActivationInt;
file >> hiddenActivationInt >> outputActivationInt;

// Convert integers to enum values
ActivationFunction newHiddenActivation = static_cast<ActivationFunction>(hiddenActivationInt);
ActivationFunction newOutputActivation = static_cast<ActivationFunction>(outputActivationInt);

// Initialize the network with the loaded architecture
layerSizes = newLayerSizes;
hiddenActivation = newHiddenActivation;
outputActivation = newOutputActivation;

// Reset weights and biases
weights.clear();
biases.clear();

for (size_t i = 1; i < layerSizes.size(); i++) {
weights.push_back(std::vector<std::vector<double>>(
layerSizes[i], std::vector<double>(layerSizes[i-1])));
biases.push_back(std::vector<double>(layerSizes[i], 0.0));
}

// Read weights
for (size_t i = 0; i < weights.size(); i++) {
for (size_t j = 0; j < weights[i].size(); j++) {
for (size_t k = 0; k < weights[i][j].size(); k++) {
file >> weights[i][j][k];
}
}
}

// Read biases
for (size_t i = 0; i < biases.size(); i++) {
for (size_t j = 0; j < biases[i].size(); j++) {
file >> biases[i][j];
}
}

// Initialize optimizer state variables
velocities.resize(layerSizes.size() - 1);
cacheWeights.resize(layerSizes.size() - 1);
velocitiesBias.resize(layerSizes.size() - 1);
cacheBias.resize(layerSizes.size() - 1);

for (size_t i = 0; i < weights.size(); i++) {
velocities[i].resize(weights[i].size());
cacheWeights[i].resize(weights[i].size());

for (size_t j = 0; j < weights[i].size(); j++) {
velocities[i][j].resize(weights[i][j].size(), 0.0);
cacheWeights[i][j].resize(weights[i][j].size(), 0.0);
}

velocitiesBias[i].resize(biases[i].size(), 0.0);
cacheBias[i].resize(biases[i].size(), 0.0);
}

file.close();
}

std::string NeuralNetwork::getModelSummary() const {
std::stringstream ss;

ss << "Neural Network Summary:" << std::endl;
ss << "======================" << std::endl;
ss << "Architecture: ";
for (size_t i = 0; i < layerSizes.size(); i++) {
ss << layerSizes[i];
if (i < layerSizes.size() - 1) ss << " -> ";
}
ss << std::endl;

ss << "Hidden Activation: ";
switch (hiddenActivation) {
case ActivationFunction::SIGMOID: ss << "Sigmoid"; break;
case ActivationFunction::RELU: ss << "ReLU"; break;
case ActivationFunction::LEAKY_RELU: ss << "Leaky ReLU"; break;
case ActivationFunction::TANH: ss << "Tanh"; break;
case ActivationFunction::LINEAR: ss << "Linear"; break;
default: ss << "Unknown";
}
ss << std::endl;

ss << "Output Activation: ";
switch (outputActivation) {
case ActivationFunction::SIGMOID: ss << "Sigmoid"; break;
case ActivationFunction::RELU: ss << "ReLU"; break;
case ActivationFunction::LEAKY_RELU: ss << "Leaky ReLU"; break;
case ActivationFunction::TANH: ss << "Tanh"; break;
case ActivationFunction::LINEAR: ss << "Linear"; break;
case ActivationFunction::SOFTMAX: ss << "Softmax"; break;
default: ss << "Unknown";
}
ss << std::endl;

// Count parameters
int totalParams = 0;
for (size_t i = 0; i < weights.size(); i++) {
for (size_t j = 0; j < weights[i].size(); j++) {
totalParams += weights[i][j].size();
}
totalParams += biases[i].size();
}

ss << "Total Parameters: " << totalParams << std::endl;

return ss.str();
}