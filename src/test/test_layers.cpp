#include <gtest/gtest.h>
#include "layer.h"
#include "dense_layer.h"
#include "dropout_layer.h"
#include "batch_norm_layer.h"

class LayerTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code here, if needed
    }

    void TearDown() override {
        // Cleanup code here, if needed
    }
};

TEST_F(LayerTests, DenseLayerForward) {
    DenseLayer layer(3, 2); // 3 inputs, 2 outputs
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = layer.forward(input, true);
    
    ASSERT_EQ(output.size(), 2);
    // Add more assertions to check the output values based on initialized weights and biases
}

TEST_F(LayerTests, DenseLayerBackward) {
    DenseLayer layer(3, 2);
    std::vector<double> outputGradient = {0.5, -0.5};
    std::vector<double> inputGradient = layer.backward(outputGradient);
    
    ASSERT_EQ(inputGradient.size(), 3);
    // Add more assertions to check the input gradient values
}

TEST_F(LayerTests, DropoutLayerForward) {
    DropoutLayer layer(0.5); // 50% dropout rate
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = layer.forward(input, true);
    
    ASSERT_EQ(output.size(), 3);
    // Check that some outputs are zeroed out based on dropout
}

TEST_F(LayerTests, BatchNormLayerForward) {
    BatchNormLayer layer(3); // 3 inputs
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = layer.forward(input, true);
    
    ASSERT_EQ(output.size(), 3);
    // Add assertions to check the output values after batch normalization
}

// Add more tests for other layer functionalities and edge cases

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}