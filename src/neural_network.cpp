#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <memory>
#include <fstream>
#include "../core/types.h"
#include "../optimizers/optimizer.h"

enum class LayerType {
    DENSE,
    DROPOUT,
    CONV2D,
    MAXPOOLING,
    FLATTEN,
    // Add other layer types as needed
};

/**
 * @brief Abstract base class for neural network layers
 */
class Layer {
public:
    virtual ~Layer() = default;
    
    /**
     * @brief Forward pass through the layer
     * @param input Input data
     * @param training Whether the network is in training mode
     * @return Output of the layer
     */
    virtual std::vector<double> forward(const std::vector<double>& input, bool training = true) = 0;
    
    /**
     * @brief Backward pass through the layer
     * @param outputGradient Gradient from the next layer
     * @return Gradient with respect to the input
     */
    virtual std::vector<double> backward(const std::vector<double>& outputGradient) = 0;
    
    /**
     * @brief Update layer parameters using the optimizer
     * @param optimizer Optimizer to use
     * @param iteration Current iteration number
     */
    virtual void updateParameters(Optimizer& optimizer, int iteration) = 0;
    
    /**
     * @brief Get the number of parameters in the layer
     * @return Number of trainable parameters
     */
    virtual size_t getParameterCount() const = 0;
    
    /**
     * @brief Get the output of the layer from the last forward pass
     * @return Layer output
     */
    virtual std::vector<double> getOutput() const = 0;
    
    /**
     * @brief Get the input gradient calculated during the last backward pass
     * @return Input gradient
     */
    virtual std::vector<double> getInputGradient() const = 0;
    
    /**
     * @brief Save the layer parameters to a file
     * @param file Output file stream
     */
    virtual void save(std::ofstream& file) const = 0;
    
    /**
     * @brief Load layer parameters from a file
     * @param file Input file stream
     */
    virtual void load(std::ifstream& file) = 0;
    
    /**
     * @brief Create a layer of the specified type
     * @param type Type of layer to create
     * @return Unique pointer to the created layer
     */
    static std::unique_ptr<Layer> create(LayerType type);
};

#endif // LAYER_H