// filepath: /cpp_neural_network/cpp_neural_network/src/optimizers/sgd.h

#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

class SGD : public Optimizer {
private:
    double learningRate;

public:
    SGD(double learningRate);
    
    void update(std::vector<double>& weights, 
                const std::vector<double>& gradients,
                int iteration) override;
    
    std::unique_ptr<Optimizer> clone() const override;
    
    void reset() override;
    
    OptimizerType getType() const override;
};

#endif // SGD_H