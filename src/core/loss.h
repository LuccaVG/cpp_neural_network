// filepath: /cpp_neural_network/cpp_neural_network/src/core/loss.h

#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <stdexcept>

enum class LossType {
    MEAN_SQUARED_ERROR,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY,
    HUBER_LOSS,
    MEAN_ABSOLUTE_ERROR
};

class Loss {
public:
    virtual ~Loss() = default;

    virtual double calculate(const std::vector<double>& predicted, 
                             const std::vector<double>& target) const = 0;

    virtual std::vector<double> gradient(const std::vector<double>& predicted, 
                                         const std::vector<double>& target) const = 0;

    static std::unique_ptr<Loss> create(LossType type);
};

#endif // LOSS_H