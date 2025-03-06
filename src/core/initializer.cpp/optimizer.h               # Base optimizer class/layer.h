#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <memory>
#include <stdexcept>

enum class OptimizerType {
    SGD,
    MOMENTUM,
    RMSPROP,
    ADAM
};

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void update(std::vector<double>& weights, 
                        const std::vector<double>& gradients,
                        int iteration) = 0;

    virtual std::unique_ptr<Optimizer> clone() const = 0;

    virtual void reset() = 0;

    virtual OptimizerType getType() const = 0;

    static std::unique_ptr<Optimizer> create(OptimizerType type, double learningRate);
};

#endif // OPTIMIZER_H