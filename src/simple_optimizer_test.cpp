#include <iostream>
#include <vector>
#include "optimizers/optimizer.h"

int main() {
    std::cout << "Testing optimizers after cleanup..." << std::endl;
    
    // Test weights and gradients
    std::vector<double> weights = {1.0, 2.0, 3.0};
    std::vector<double> gradients = {0.1, 0.2, 0.3};
    double learningRate = 0.01;
    
    std::cout << "Initial weights: ";
    for (double w : weights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    
    // Test SGD
    SGD sgd;
    std::vector<double> sgdWeights = weights;
    sgd.update(sgdWeights, gradients, learningRate);
    std::cout << "SGD updated weights: ";
    for (double w : sgdWeights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    
    // Test Adam
    Adam adam;
    std::vector<double> adamWeights = weights;
    adam.update(adamWeights, gradients, learningRate);
    std::cout << "Adam updated weights: ";
    for (double w : adamWeights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    
    // Test Momentum
    Momentum momentum;
    std::vector<double> momentumWeights = weights;
    momentum.update(momentumWeights, gradients, learningRate);
    std::cout << "Momentum updated weights: ";
    for (double w : momentumWeights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    
    // Test RMSProp
    RMSProp rmsprop;
    std::vector<double> rmspropWeights = weights;
    rmsprop.update(rmspropWeights, gradients, learningRate);
    std::cout << "RMSProp updated weights: ";
    for (double w : rmspropWeights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    
    std::cout << "All optimizers tested successfully!" << std::endl;
    return 0;
}
