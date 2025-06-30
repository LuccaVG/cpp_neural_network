#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "types.h"
#include <vector>
#include <memory>
#include <cmath>

class Activation {
public:
    virtual ~Activation() = default;
    
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual std::vector<double> activate(const std::vector<double>& inputs) const;
    virtual std::vector<double> derivative(const std::vector<double>& inputs) const;
    
    static std::unique_ptr<Activation> create(ActivationType type, double param = 0.0);
    
    // Static utility functions for compatibility
    static double apply(double x, ActivationType type);
    static std::vector<double> apply(const std::vector<double>& x, ActivationType type);
    static double derivative(double x, ActivationType type);
    static std::vector<double> derivative(const std::vector<double>& x, ActivationType type);
    
    // Advanced activation functions
    static double leakyRelu(double x, double alpha = 0.01);
    static double leakyReluDerivative(double x, double alpha = 0.01);
    
    static double elu(double x, double alpha = 1.0);
    static double eluDerivative(double x, double alpha = 1.0);
    
    static double swish(double x);
    static double swishDerivative(double x);
    
    static double gelu(double x);
    static double geluDerivative(double x);
    
    static double mish(double x);
    static double mishDerivative(double x);
    
    // Parametric activation functions
    static std::vector<double> softmax(const std::vector<double>& x);
    static std::vector<double> softmaxDerivative(const std::vector<double>& x, const std::vector<double>& outputGradient);
};

// Linear activation
class Linear : public Activation {
public:
    double activate(double x) const override { return x; }
    double derivative(double x) const override { return 1.0; }
};

// Sigmoid activation
class Sigmoid : public Activation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

// Tanh activation
class Tanh : public Activation {
public:
    double activate(double x) const override { return std::tanh(x); }
    double derivative(double x) const override;
};

// ReLU activation
class ReLU : public Activation {
public:
    double activate(double x) const override { return std::max(0.0, x); }
    double derivative(double x) const override { return x > 0 ? 1.0 : 0.0; }
};

// Leaky ReLU activation
class LeakyReLU : public Activation {
private:
    double alpha;
public:
    LeakyReLU(double alpha = 0.01) : alpha(alpha) {}
    double activate(double x) const override { return x > 0 ? x : alpha * x; }
    double derivative(double x) const override { return x > 0 ? 1.0 : alpha; }
};

// ELU activation
class ELU : public Activation {
private:
    double alpha;
public:
    ELU(double alpha = 1.0) : alpha(alpha) {}
    double activate(double x) const override;
    double derivative(double x) const override;
};

// Swish activation
class Swish : public Activation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

// GELU activation
class GELU : public Activation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

// SELU activation
class SELU : public Activation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

// Mish activation
class Mish : public Activation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

// Softmax activation (special case - operates on vectors)
class Softmax : public Activation {
public:
    double activate(double x) const override { return x; } // Not used for softmax
    double derivative(double x) const override { return 1.0; } // Not used for softmax
    std::vector<double> activate(const std::vector<double>& inputs) const override;
    std::vector<double> derivative(const std::vector<double>& inputs) const override;
};

#endif // ACTIVATION_H