#include "neuron.h"

Neuron::Neuron(int numInputs) {
    weights.resize(numInputs);
    std::generate(weights.begin(), weights.end(), []() { return ((double) rand() / (RAND_MAX)) * 2 - 1; });
    bias = ((double) rand() / (RAND_MAX)) * 2 - 1;
    output = 0.0;
    delta = 0.0;
}

double Neuron::activate(const std::vector<double>& inputs) {
    double activation = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        activation += weights[i] * inputs[i];
    }
    output = sigmoid(activation);
    return output;
}

double Neuron::getOutput() const {
    return output;
}

void Neuron::setWeights(const std::vector<double>& newWeights) {
    weights = newWeights;
}

std::vector<double> Neuron::getWeights() const {
    return weights;
}

void Neuron::setDelta(double delta) {
    this->delta = delta;
}

double Neuron::getDelta() const {
    return delta;
}

void Neuron::updateWeights(const std::vector<double>& inputs, double learningRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += learningRate * delta * inputs[i];
    }
    bias += learningRate * delta;
}

double Neuron::sigmoid(double x) const {
    return 1.0 / (1.0 + exp(-x));
}

double Neuron::sigmoidDerivative(double x) const {
    return x * (1.0 - x);
}