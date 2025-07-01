#include <gtest/gtest.h>
#include "optimizer.h"

class OptimizerTests : public ::testing::Test {
protected:
    std::vector<double> weights;
    std::vector<double> gradients;

    void SetUp() override {
        weights = {0.5, -0.5, 0.0};
        gradients = {0.1, -0.1, 0.2};
    }
};

TEST_F(OptimizerTests, SGDUpdate) {
    SGD sgd(0.01);
    std::vector<double> initialWeights = weights;

    sgd.update(weights, gradients, 1);

    EXPECT_NE(weights[0], initialWeights[0]);
    EXPECT_NE(weights[1], initialWeights[1]);
    EXPECT_NE(weights[2], initialWeights[2]);
}

TEST_F(OptimizerTests, MomentumUpdate) {
    Momentum momentum(0.01, 0.9);
    std::vector<double> initialWeights = weights;

    momentum.update(weights, gradients, 1);

    EXPECT_NE(weights[0], initialWeights[0]);
    EXPECT_NE(weights[1], initialWeights[1]);
    EXPECT_NE(weights[2], initialWeights[2]);
}

TEST_F(OptimizerTests, RMSPropUpdate) {
    RMSProp rmsprop(0.01);
    std::vector<double> initialWeights = weights;

    rmsprop.update(weights, gradients, 1);

    EXPECT_NE(weights[0], initialWeights[0]);
    EXPECT_NE(weights[1], initialWeights[1]);
    EXPECT_NE(weights[2], initialWeights[2]);
}

TEST_F(OptimizerTests, AdamUpdate) {
    Adam adam(0.01);
    std::vector<double> initialWeights = weights;

    adam.update(weights, gradients, 1);

    EXPECT_NE(weights[0], initialWeights[0]);
    EXPECT_NE(weights[1], initialWeights[1]);
    EXPECT_NE(weights[2], initialWeights[2]);
}