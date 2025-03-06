#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <memory>
#include "types.h"

class Loss {
public:
    Loss(LossType type) : type(type) {}
    virtual ~Loss() = default;

    // Member variables
    LossType type;

    // Make calculate a const method to match implementation in loss.cpp
    virtual double calculate(const std::vector<double>& predicted, const std::vector<double>& target) const = 0;
    virtual std::vector<double> calculateGradient(const std::vector<double>& predicted, const std::vector<double>& target) const = 0;
    
    // Add gradient method that was missing
    virtual std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& target) const = 0;

    // Factory method
    static std::unique_ptr<Loss> create(LossType type);
};

class MeanSquaredError : public Loss {
public:
    MeanSquaredError() : Loss(LossType::MEAN_SQUARED_ERROR) {}

    double calculate(const std::vector<double>& predicted, const std::vector<double>& target) const override;
    std::vector<double> calculateGradient(const std::vector<double>& predicted, const std::vector<double>& target) const override;
    std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& target) const override;
};

class BinaryCrossEntropy : public Loss {
public:
    BinaryCrossEntropy() : Loss(LossType::BINARY_CROSS_ENTROPY) {}

    double calculate(const std::vector<double>& predicted, const std::vector<double>& target) const override;
    std::vector<double> calculateGradient(const std::vector<double>& predicted, const std::vector<double>& target) const override;
    std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& target) const override;
};

// Add other loss classes as needed

#endif // LOSS_H