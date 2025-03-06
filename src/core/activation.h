// filepath: /cpp_neural_network/cpp_neural_network/src/core/activation.h

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include "types.h"

class Activation {
public:
    static double apply(double x, ActivationType type);
    static std::vector<double> apply(const std::vector<double>& x, ActivationType type);
    static double derivative(double x, ActivationType type);
    static std::vector<double> derivative(const std::vector<double>& x, ActivationType type);
};

#endif // ACTIVATION_H