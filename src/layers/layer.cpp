#include "layer.h"
#include "dense_layer.h"
#include "dropout_layer.h"
#include "batch_norm_layer.h"
#include <stdexcept>

/**
 * @brief Base class for neural network layers
 */
Layer::~Layer() = default;

/**
 * @brief Create layer from type
 * @param type Layer type
 * @param inputSize Input size
 * @param outputSize Output size
 * @param activation Activation function
 * @return Unique pointer to layer
 */
std::unique_ptr<Layer> Layer::create(LayerType type, size_t inputSize, size_t outputSize, ActivationType activation) {
    switch (type) {
        case LayerType::DENSE:
            return std::make_unique<DenseLayer>(inputSize, outputSize, activation);
        case LayerType::DROPOUT:
            return std::make_unique<DropoutLayer>(inputSize);
        case LayerType::BATCH_NORMALIZATION:
            return std::make_unique<BatchNormLayer>(inputSize);
        default:
            throw std::runtime_error("Unsupported layer type");
    }
}