#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include <random>
#include <vector>

class DropoutLayer : public Layer {
private:
    double dropoutRate;
    std::vector<bool> mask;
    std::vector<double> lastOutput;
    std::vector<double> inputGradient;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
    
public:
    DropoutLayer(double dropoutRate = 0.5) 
        : dropoutRate(dropoutRate), rng(std::random_device{}()), dist(0.0, 1.0) {}
    
    void forward(const std::vector<double>& input) override {
        mask.resize(input.size());
        lastOutput.resize(input.size());
        
        // During training, randomly set some neurons to zero
        for (size_t i = 0; i < input.size(); ++i) {
            if (dist(rng) < dropoutRate) {
                mask[i] = false;
                lastOutput[i] = 0.0;
            } else {
                mask[i] = true;
                // Scale by 1/(1-dropoutRate) to maintain expected value
                lastOutput[i] = input[i] / (1.0 - dropoutRate);
            }
        }
    }
    
    std::vector<double> backward(const std::vector<double>& outputGradient) override {
        inputGradient.resize(outputGradient.size());
        
        for (size_t i = 0; i < outputGradient.size(); ++i) {
            if (mask[i]) {
                inputGradient[i] = outputGradient[i] / (1.0 - dropoutRate);
            } else {
                inputGradient[i] = 0.0;
            }
        }
        
        return inputGradient;
    }
    
    void updateParameters(Optimizer& optimizer, int iteration) override {
        // Dropout layer has no parameters to update
    }
    
    std::vector<double> getOutput() const override {
        return lastOutput;
    }
    
    std::vector<double> getInputGradient() const override {
        return inputGradient;
    }
    
    void save(std::ofstream& file) const override {
        file.write(reinterpret_cast<const char*>(&dropoutRate), sizeof(dropoutRate));
    }
    
    void load(std::ifstream& file) override {
        file.read(reinterpret_cast<char*>(&dropoutRate), sizeof(dropoutRate));
    }
    
    void setTraining(bool training) {
        // In inference mode, don't apply dropout
        if (!training) {
            dropoutRate = 0.0;
        }
    }
};

#endif // DROPOUT_LAYER_H
