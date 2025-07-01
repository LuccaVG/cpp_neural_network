#include <gtest/gtest.h>
#include "../src/neural_network.h"
#include "../src/core/activation.h"
#include "../src/core/loss.h"
#include "../src/optimizers/optimizer.h"
#include "../src/layers/dense_layer.h"

class NeuralNetworkTest : public ::testing::Test {
protected:
    NeuralNetwork nn;

    void SetUp() override {
        nn.addLayer(std::make_unique<DenseLayer>(2, 3, ActivationType::RELU));
        nn.addLayer(std::make_unique<DenseLayer>(3, 1, ActivationType::SIGMOID));
    }
};

TEST_F(NeuralNetworkTest, ForwardPass) {
    std::vector<double> input = {0.5, 0.2};
    std::vector<double> output = nn.forward(input);
    
    ASSERT_EQ(output.size(), 1);
}

TEST_F(NeuralNetworkTest, BackwardPass) {
    std::vector<double> input = {0.5, 0.2};
    std::vector<double> target = {1.0};
    
    nn.forward(input);
    nn.backward(target);
    
    ASSERT_NO_THROW(nn.updateParameters(OptimizerType::ADAM, 0.01));
}

TEST_F(NeuralNetworkTest, LossCalculation) {
    std::vector<double> predicted = {0.8};
    std::vector<double> target = {1.0};
    
    double loss = Loss::calculate(predicted, target, LossType::BINARY_CROSS_ENTROPY);
    
    ASSERT_GT(loss, 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}