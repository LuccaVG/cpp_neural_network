#include <gtest/gtest.h>
#include "core/activation.h"

class ActivationTest : public ::testing::Test {
protected:
    Activation activation;

    void SetUp() override {
        // Any setup code can go here
    }
};

TEST_F(ActivationTest, TestSigmoid) {
    EXPECT_NEAR(activation.apply(0.0, ActivationType::SIGMOID), 0.5, 1e-5);
    EXPECT_NEAR(activation.apply(1.0, ActivationType::SIGMOID), 0.731058, 1e-5);
    EXPECT_NEAR(activation.apply(-1.0, ActivationType::SIGMOID), 0.268941, 1e-5);
}

TEST_F(ActivationTest, TestReLU) {
    EXPECT_EQ(activation.apply(0.0, ActivationType::RELU), 0.0);
    EXPECT_EQ(activation.apply(1.0, ActivationType::RELU), 1.0);
    EXPECT_EQ(activation.apply(-1.0, ActivationType::RELU), 0.0);
}

TEST_F(ActivationTest, TestTanh) {
    EXPECT_NEAR(activation.apply(0.0, ActivationType::TANH), 0.0, 1e-5);
    EXPECT_NEAR(activation.apply(1.0, ActivationType::TANH), 0.761594, 1e-5);
    EXPECT_NEAR(activation.apply(-1.0, ActivationType::TANH), -0.761594, 1e-5);
}

TEST_F(ActivationTest, TestSoftmax) {
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = activation.apply(input, ActivationType::SOFTMAX);
    EXPECT_NEAR(output[0], 0.09003057, 1e-5);
    EXPECT_NEAR(output[1], 0.24472847, 1e-5);
    EXPECT_NEAR(output[2], 0.66524096, 1e-5);
}

TEST_F(ActivationTest, TestLeakyReLU) {
    EXPECT_EQ(activation.apply(0.0, ActivationType::LEAKY_RELU), 0.0);
    EXPECT_EQ(activation.apply(1.0, ActivationType::LEAKY_RELU), 1.0);
    EXPECT_EQ(activation.apply(-1.0, ActivationType::LEAKY_RELU), -0.01);
}

TEST_F(ActivationTest, TestELU) {
    EXPECT_NEAR(activation.apply(0.0, ActivationType::ELU), 0.0, 1e-5);
    EXPECT_NEAR(activation.apply(1.0, ActivationType::ELU), 1.0, 1e-5);
    EXPECT_NEAR(activation.apply(-1.0, ActivationType::ELU), std::exp(-1.0) - 1.0, 1e-5);
}

TEST_F(ActivationTest, TestSwish) {
    EXPECT_NEAR(activation.apply(0.0, ActivationType::SWISH), 0.0, 1e-5);
    EXPECT_NEAR(activation.apply(1.0, ActivationType::SWISH), 0.731058, 1e-5);
    EXPECT_NEAR(activation.apply(-1.0, ActivationType::SWISH), -0.268941, 1e-5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}