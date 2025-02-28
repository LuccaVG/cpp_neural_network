#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "neuron.h"

class NeuralLayer {
public:
    NeuralLayer(int numNeurons, int numInputsPerNeuron);
    std::vector<double> feedForward(const std::vector<double>& inputs);
    std::vector<Neuron>& getNeurons();

private:
    std::vector<Neuron> neurons;
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology);
    std::vector<double> feedForward(const std::vector<double>& inputs);
    void train(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate);

private:
    std::vector<NeuralLayer> layers;
};

#endif // NEURAL_NETWORK_H