#include "neural_network.h"

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