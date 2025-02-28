#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include <string>

class Neuron {
public:
    Neuron(int numInputs);
    double activate(const std::vector<double>& inputs);
    double getOutput() const;
    void setWeights(const std::vector<double>& newWeights);
    std::vector<double> getWeights() const;
    void setDelta(double delta);
    double getDelta() const;
    void updateWeights(const std::vector<double>& inputs, double learningRate);

private:
    std::vector<double> weights;
    double bias;
    double output;
    double delta;
    double sigmoid(double x) const;

public:
    double sigmoidDerivative(double x) const;
};

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

class Memory {
public:
    void store(const std::string& key, const std::vector<double>& value);
    std::vector<double> retrieve(const std::string& key) const;
    bool exists(const std::string& key) const;

private:
    std::unordered_map<std::string, std::vector<double>> memory;
};

bool Memory::exists(const std::string& key) const {
    return memory.find(key) != memory.end();
}

void Memory::store(const std::string& key, const std::vector<double>& value) {
    memory[key] = value;
}

std::vector<double> Memory::retrieve(const std::string& key) const {
    auto it = memory.find(key);
    if (it != memory.end()) {
        return it->second;
    }
    return {};
}

class NeuralLayer {
public:
    NeuralLayer(int numNeurons, int numInputsPerNeuron);
    std::vector<double> feedForward(const std::vector<double>& inputs);
    std::vector<Neuron>& getNeurons();

private:
    std::vector<Neuron> neurons;
};

NeuralLayer::NeuralLayer(int numNeurons, int numInputsPerNeuron) {
    for (int i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(numInputsPerNeuron);
    }
}

std::vector<double> NeuralLayer::feedForward(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    outputs.reserve(neurons.size());
    for (auto& neuron : neurons) {
        outputs.push_back(neuron.activate(inputs));
    }
    return outputs;
}

std::vector<Neuron>& NeuralLayer::getNeurons() {
    return neurons;
}

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology);
    std::vector<double> feedForward(const std::vector<double>& inputs);
    void train(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate);

private:
    std::vector<NeuralLayer> layers;
};

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology) {
    for (size_t i = 0; i < topology.size() - 1; ++i) {
        layers.emplace_back(topology[i + 1], topology[i]);
    }
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& inputs) {
    std::vector<double> outputs = inputs;
    for (auto& layer : layers) {
        outputs = layer.feedForward(outputs);
    }
    return outputs;
}

void NeuralNetwork::train(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate) {
    std::vector<double> outputs = feedForward(inputs);
    std::vector<double> errors;

    for (size_t i = 0; i < targets.size(); ++i) {
        errors.push_back(targets[i] - outputs[i]);
    }

    for (int l = layers.size() - 1; l >= 0; --l) {
        auto& layer = layers[l].getNeurons();
        std::vector<double> nextErrors(layer[0].getWeights().size(), 0.0);

        for (size_t n = 0; n < layer.size(); ++n) {
            double delta = errors[n] * layer[n].sigmoidDerivative(layer[n].getOutput());
            layer[n].setDelta(delta);

            for (size_t w = 0; w < layer[n].getWeights().size(); ++w) {
                nextErrors[w] += layer[n].getWeights()[w] * delta;
            }
            layer[n].updateWeights(inputs, learningRate);
        }
        errors = nextErrors;
    }
}

void feedInformation(NeuralNetwork& nn, Memory& memory) {
    std::string key;
    std::vector<double> inputs(3);
    std::vector<double> outputs;

    std::cout << "Enter a key to store information: ";
    std::cin >> key;

    std::cout << "Enter 3 inputs: ";
    for (double& input : inputs) {
        std::cin >> input;
    }

    outputs = nn.feedForward(inputs);
    memory.store(key + "_inputs", inputs);
    memory.store(key + "_outputs", outputs);

    std::cout << "Information stored under key: " << key << std::endl;
}

int main() {
    std::srand(std::time(nullptr));
    NeuralNetwork nn({3, 5, 3, 1});
    Memory memory;

    std::cout << "Starting AI system...\n";
    feedInformation(nn, memory);

    return 0;
}
