#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <memory>
#include <string>
#include <fstream>

// Forward declarations
class Optimizer;

/**
 * @brief Enumeration of supported activation functions
 */
enum class ActivationType {
    LINEAR,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    ELU,
    SOFTMAX,
    SWISH
};

/**
 * @brief Enumeration of supported layer types
 */
enum class LayerType {
    INPUT,
    DENSE,
    DROPOUT,
    BATCH_NORMALIZATION,
    CONVOLUTIONAL,
    MAX_POOLING,
    FLATTEN,
    LSTM,
    GRU
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
    virtual std::vector<double> forward(const std::vector<double>& input, bool training) = 0;
    
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
     * @brief Get the output shape of the layer
     * @return Output size
     */
    virtual size_t getOutputSize() const = 0;
    
    /**
     * @brief Get the layer type
     * @return Layer type
     */
    virtual LayerType getType() const = 0;
    
    /**
     * @brief Get layer name
     * @return Layer name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Save layer parameters to file
     * @param file Output file stream
     */
    virtual void saveParameters(std::ofstream& file) const = 0;
    
    /**
     * @brief Load layer parameters from file
     * @param file Input file stream
     */
    virtual void loadParameters(std::ifstream& file) = 0;
    
    /**
     * @brief Reset layer state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Create layer from type
     * @param type Layer type
     * @param inputSize Input size
     * @param outputSize Output size
     * @param activation Activation function
     * @return Unique pointer to layer
     */
    static std::unique_ptr<Layer> create(LayerType type, 
                                         size_t inputSize,
                                         size_t outputSize,
                                         ActivationType activation = ActivationType::SIGMOID);
};

// Forward declarations for concrete layer classes used in the factory method
class DenseLayer;
class DropoutLayer;
class BatchNormLayer;

#endif // LAYER_H