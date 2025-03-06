// filepath: /cpp_neural_network/cpp_neural_network/src/utils/random.h

#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <vector>

class Random {
public:
    static void seed(unsigned int seedValue);
    static double uniform(double lower, double upper);
    static std::vector<double> uniformVector(size_t size, double lower, double upper);
    static double normal(double mean, double stddev);
    static std::vector<double> normalVector(size_t size, double mean, double stddev);
};

#endif // RANDOM_H