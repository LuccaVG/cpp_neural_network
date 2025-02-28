#pragma once

#include <vector>
#include <random>
#include <functional>
#include <memory>
#include <iostream>
#include <cmath>

class NeuralLayer {
private:
    std::vector<std::vector<double>> weights;  // Weights connecting to the previous layer
    std::vector<double> biases;                // Bias values for each neuron
    std::vector<double> outputs;               // Output values for each neuron
    std::vector<double> errors;                // Error values for backpropagation
    std::function<double(double)> activation;  // Activation function
    std::function<double(double)> activation_derivative; // Derivative of activation function
    
    // Random number generator for weight initialization
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

public:
    // Constructor for input layer (no weights or biases)
    NeuralLayer(int size);
    
    // Constructor for hidden and output layers
    NeuralLayer(int size, int prev_layer_size,
                std::function<double(double)> activation_func,
                std::function<double(double)> activation_derivative_func);
    
    // Forward propagate inputs through this layer
    std::vector<double> forward(const std::vector<double>& inputs);
    
    // Backpropagation for output layer
    void calculate_output_error(const std::vector<double>& targets);
    
    // Backpropagation for hidden layers
    void calculate_hidden_error(const NeuralLayer& next_layer);
    
    // Update weights and biases based on errors
    void update_weights(const std::vector<double>& prev_outputs, double learning_rate);
    
    // Getters
    const std::vector<double>& get_outputs() const { return outputs; }
    const std::vector<std::vector<double>>& get_weights() const { return weights; }
    const std::vector<double>& get_biases() const { return biases; }
    const std::vector<double>& get_errors() const { return errors; }
    int get_size() const { return outputs.size(); }
    
    // For debugging
    void print_details() const;
};

// Common activation functions
namespace ActivationFunctions {
    // Sigmoid activation
    inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    inline double sigmoid_derivative(double x) { return x * (1.0 - x); }
    
    // ReLU activation
    inline double relu(double x) { return std::max(0.0, x); }
    inline double relu_derivative(double x) { return x > 0.0 ? 1.0 : 0.0; }
    
    // Tanh activation
    inline double tanh_activation(double x) { return tanh(x); }
    inline double tanh_derivative(double x) { return 1.0 - x * x; }
    
    // Linear activation
    inline double linear(double x) { return x; }
    inline double linear_derivative(double x) { return 1.0; }
}