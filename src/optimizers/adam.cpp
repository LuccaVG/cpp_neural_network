#include "adam.h"
#include <cmath>
#include <algorithm>

Adam::Adam(double beta1, double beta2, double epsilon)
    : beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

void Adam::update(std::vector<double>& weights, const std::vector<double>& gradients, double learningRate) {
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