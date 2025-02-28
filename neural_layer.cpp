#include "neural_layer.h"
#include <ctime>
#include <algorithm>

// Constructor for input layer (no weights or biases)
NeuralLayer::NeuralLayer(int size) 
    : outputs(size, 0.0), 
      activation([](double x) { return x; }),  // Identity function
      activation_derivative([](double x) { return 1.0; }) {
    
    // Initialize RNG
    std::random_device rd;
    rng = std::mt19937(rd());
    dist = std::uniform_real_distribution<double>(-1.0, 1.0);
}

// Constructor for hidden and output layers
NeuralLayer::NeuralLayer(int size, int prev_layer_size,
                         std::function<double(double)> activation_func,
                         std::function<double(double)> activation_derivative_func) 
    : biases(size, 0.0), 
      outputs(size, 0.0), 
      errors(size, 0.0),
      activation(activation_func),
      activation_derivative(activation_derivative_func) {
    
    // Initialize RNG
    std::random_device rd;
    rng = std::mt19937(rd());
    dist = std::uniform_real_distribution<double>(-1.0, 1.0);
    
    // Initialize weights with random values
    weights.resize(size);
    for (auto& neuron_weights : weights) {
        neuron_weights.resize(prev_layer_size);
        for (auto& weight : neuron_weights) {
            weight = dist(rng) * sqrt(1.0 / prev_layer_size);  // He initialization
        }
    }
    
    // Initialize biases with small random values
    for (auto& bias : biases) {
        bias = dist(rng) * 0.1;
    }
}

// Forward propagate inputs through this layer
std::vector<double> NeuralLayer::forward(const std::vector<double>& inputs) {
    // For input layer, just copy the inputs to outputs
    if (weights.empty()) {
        outputs = inputs;
        return outputs;
    }
    
    // For other layers, compute weighted sum and apply activation function
    for (size_t i = 0; i < outputs.size(); ++i) {
        double sum = biases[i];
        for (size_t j = 0; j < inputs.size(); ++j) {
            sum += weights[i][j] * inputs[j];
        }
        outputs[i] = activation(sum);
    }
    
    return outputs;
}

// Backpropagation for output layer
void NeuralLayer::calculate_output_error(const std::vector<double>& targets) {
    for (size_t i = 0; i < outputs.size(); ++i) {
        // Error = derivative of activation function * difference between target and output
        errors[i] = activation_derivative(outputs[i]) * (targets[i] - outputs[i]);
    }
}

// Backpropagation for hidden layers
void NeuralLayer::calculate_hidden_error(const NeuralLayer& next_layer) {
    for (size_t i = 0; i < outputs.size(); ++i) {
        double error_sum = 0.0;
        const auto& next_weights = next_layer.get_weights();
        const auto& next_errors = next_layer.get_errors();
        
        // Sum errors from all connected neurons in the next layer
        for (size_t j = 0; j < next_errors.size(); ++j) {
            error_sum += next_weights[j][i] * next_errors[j];
        }
        
        errors[i] = activation_derivative(outputs[i]) * error_sum;
    }
}

// Update weights and biases based on errors
void NeuralLayer::update_weights(const std::vector<double>& prev_outputs, double learning_rate) {
    // Skip if this is an input layer
    if (weights.empty()) return;
    
    // Update weights
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < prev_outputs.size(); ++j) {
            weights[i][j] += learning_rate * errors[i] * prev_outputs[j];
        }
        // Update bias
        biases[i] += learning_rate * errors[i];
    }
}

// Print layer details for debugging
void NeuralLayer::print_details() const {
    std::cout << "Layer size: " << outputs.size() << std::endl;
    
    if (!weights.empty()) {
        std::cout << "Weights:" << std::endl;
        for (size_t i = 0; i < weights.size(); ++i) {
            std::cout << "  Neuron " << i << ": ";
            for (size_t j = 0; j < weights[i].size(); ++j) {
                std::cout << weights[i][j] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Biases: ";
        for (const auto& bias : biases) {
            std::cout << bias << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Outputs: ";
    for (const auto& output : outputs) {
        std::cout << output << " ";
    }
    std::cout << std::endl;
}