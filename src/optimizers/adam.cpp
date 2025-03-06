#include "adam.h"
#include "optimizer.h"
#include <cmath>
#include <algorithm>

Adam::Adam(double learningRate, double beta1, double beta2, double epsilon)
    : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

void Adam::update(std::vector<double>& weights, const std::vector<double>& gradients, int iteration) {
    if (m.size() != weights.size()) {
        m.resize(weights.size(), 0.0);
        v.resize(weights.size(), 0.0);
    }

    t++;

    for (size_t i = 0; i < weights.size(); ++i) {
        m[i] = beta1 * m[i] + (1.0 - beta1) * gradients[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * gradients[i] * gradients[i];

        double m_hat = m[i] / (1.0 - std::pow(beta1, t));
        double v_hat = v[i] / (1.0 - std::pow(beta2, t));

        weights[i] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

std::unique_ptr<Optimizer> Adam::clone() const {
    auto clone = std::make_unique<Adam>(learningRate, beta1, beta2, epsilon);
    clone->m = m;
    clone->v = v;
    clone->t = t;
    return clone;
}

void Adam::reset() {
    std::fill(m.begin(), m.end(), 0.0);
    std::fill(v.begin(), v.end(), 0.0);
    t = 0;
}

OptimizerType Adam::getType() const {
    return OptimizerType::ADAM;
}