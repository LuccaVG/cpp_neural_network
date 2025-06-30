#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <memory>
#include <fstream>
#include "../core/types.h"
#include "../optimizers/optimizer.h"

class Layer {
public:
    virtual ~Layer() = default;
    
    virtual void forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& outputGradient) = 0;
    virtual void updateParameters(Optimizer& optimizer, int iteration) = 0;
    
    virtual std::vector<double> getOutput() const = 0;
    virtual std::vector<double> getInputGradient() const = 0;
    virtual void save(std::ofstream& file) const = 0;
    virtual void load(std::ifstream& file) = 0;
    
    // Factory method for layer creation during loading
    static std::unique_ptr<Layer> create(LayerType type);
};

#endif // LAYER_H