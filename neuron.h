#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

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

#endif // NEURON_H