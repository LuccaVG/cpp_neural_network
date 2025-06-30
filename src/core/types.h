#ifndef TYPES_H
#define TYPES_H

#include <cstddef>
#include <string>
#include <vector>

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
    SWISH,
    GELU,
    SELU,
    MISH,
    HARD_SIGMOID,
    HARD_TANH,
    EXPONENTIAL,
    SOFTPLUS,
    SOFTSIGN
};

/**
 * @brief Enumeration of supported loss functions
 */
enum class LossType {
    MEAN_SQUARED_ERROR,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY,
    HUBER_LOSS,
    MEAN_ABSOLUTE_ERROR,
    SPARSE_CATEGORICAL_CROSS_ENTROPY,
    FOCAL_LOSS,
    HINGE_LOSS
};

/**
 * @brief Enumeration of supported optimizer types
 */
enum class OptimizerType {
    SGD,
    MOMENTUM,
    RMSPROP,
    ADAM,
    ADAGRAD,
    ADADELTA,
    ADAMAX,
    NADAM
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
 * @brief Enumeration of weight initialization methods
 */
enum class InitializerType {
    ZEROS,
    ONES,
    RANDOM_UNIFORM,
    RANDOM_NORMAL,
    GLOROT_UNIFORM,
    GLOROT_NORMAL,
    HE_UNIFORM,
    HE_NORMAL
};

#endif // TYPES_H