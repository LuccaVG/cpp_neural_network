// filepath: /cpp_neural_network/cpp_neural_network/src/core/initializer.h

#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <vector>
#include <random>
#include <stdexcept>
#include "types.h" // Include the types header for InitializerType

class Initializer {
public:
    /**
     * @brief Initialize weights based on fan-in and fan-out
     * @param shape Shape of the weight tensor (fanIn, fanOut)
     * @param type Initialization method
     * @param rng Random number generator
     * @return Initialized weights
     */
    static std::vector<std::vector<double>> initialize(
        std::pair<size_t, size_t> shape,
        InitializerType type,
        std::mt19937& rng
    );
};

#endif // INITIALIZER_H