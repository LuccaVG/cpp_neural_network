#include "dropout_layer.h"
#include <random>
#include <algorithm>

class DropoutLayer : public Layer {
private:
    double dropoutRate;
    std::vector<double> mask;
    std::mt19937 rng;

public:
    DropoutLayer(double rate) : dropoutRate(rate) {
        std::random_device rd;
        rng.seed(rd());
    }

    std::vector<double> forward(const std::vector<double>& input, bool training) override {
        if (training) {
            mask.resize(input.size());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < input.size(); ++i) {
                mask[i] = (dist(rng) < dropoutRate) ? 0.0 : 1.0;
            }
            std::vector<double> output(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = input[i] * mask[i];
            }
            return output;
        } else {
            return input; // During inference, return input unchanged
        }
    }

    std::vector<double> backward(const std::vector<double>& outputGradient) override {
        std::vector<double> inputGradient(outputGradient.size());
        for (size_t i = 0; i < outputGradient.size(); ++i) {
            inputGradient[i] = outputGradient[i] * mask[i];
        }
        return inputGradient;
    }

    void updateParameters(Optimizer& optimizer, int iteration) override {
        // No parameters to update in dropout layer
    }

    size_t getParameterCount() const override {
        return 0; // No trainable parameters
    }

    size_t getOutputSize() const override {
        return mask.size();
    }

    LayerType getType() const override {
        return LayerType::DROPOUT;
    }

    std::string getName() const override {
        return "DropoutLayer";
    }

    void saveParameters(std::ofstream& file) const override {
        // No parameters to save
    }

    void loadParameters(std::ifstream& file) override {
        // No parameters to load
    }

    void reset() override {
        // No state to reset
    }
};