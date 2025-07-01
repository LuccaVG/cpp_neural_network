// filepath: /cpp_neural_network/cpp_neural_network/src/layers/batch_norm_layer.h

#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include "layer.h"
#include <vector>
#include <random>

class BatchNormLayer : public Layer {
private:
    size_t inputSize;
    double momentum;
    double epsilon;
    std::vector<double> gamma; // Scale parameter
    std::vector<double> beta;  // Shift parameter
    std::vector<double> runningMean; // Running mean for inference
    std::vector<double> runningVariance;  // Running variance for inference
    std::vector<double> lastInput; // Store last input for backward pass
    std::vector<double> lastOutput; // Store last output
    std::vector<double> gradGamma; // Gradients for gamma
    std::vector<double> gradBeta;  // Gradients for beta
    std::vector<double> inputGradient; // Input gradients
    bool isTraining = true;

public:
    BatchNormLayer(size_t inputSize, double momentum = 0.99, double epsilon = 1e-5);

    // Base Layer interface
    void forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient) override;
    void updateParameters(Optimizer& optimizer, int iteration) override;
    std::vector<double> getOutput() const override;
    std::vector<double> getInputGradient() const override;
    void save(std::ofstream& file) const override;
    void load(std::ifstream& file) override;
    
    // Additional methods for training/inference mode
    void setTraining(bool training) { isTraining = training; }
};

#endif // BATCH_NORM_LAYER_H