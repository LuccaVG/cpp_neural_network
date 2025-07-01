#include <iostream>
#include <vector>
#include <memory>
#include "optimizers/optimizer.h"

int main() {
    std::cout << "Testing Adam optimizer initialization..." << std::endl;
    
    try {
        // Create Adam optimizer
        auto adam = std::make_unique<Adam>();
        std::cout << "Adam optimizer created successfully" << std::endl;
        
        // Test with simple weights and gradients
        std::vector<double> weights = {1.0, 2.0, 3.0};
        std::vector<double> gradients = {0.1, 0.2, 0.3};
        
        std::cout << "Initial weights: ";
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
        
        // Perform update
        adam->update(weights, gradients, 0.01);
        
        std::cout << "Updated weights: ";
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Adam test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
