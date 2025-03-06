#include <random>
#include "random.h"

// Static engine and random device
static std::random_device rd;
static std::mt19937 eng(rd());

/**
 * @brief Generate a random double in the range [min, max].
 * @param min Minimum value.
 * @param max Maximum value.
 * @return Random double value.
 */
double Random::generateDouble(double min, double max) {
    std::uniform_real_distribution<> distr(min, max);  // Define the range
    return distr(eng);
}

/**
 * @brief Generate a random integer in the range [min, max].
 * @param min Minimum value.
 * @param max Maximum value.
 * @return Random integer value.
 */
int Random::generateInt(int min, int max) {
    std::uniform_int_distribution<> distr(min, max);  // Define the range
    return distr(eng);
}

/**
 * @brief Generate random values with normal distribution
 * @param mean Mean of the normal distribution
 * @param stddev Standard deviation of the normal distribution
 * @return Random value from normal distribution
 */
double Random::generateNormal(double mean, double stddev) {
    std::normal_distribution<> distr(mean, stddev);
    return distr(eng);
}

/**
 * @brief Initialize the random number generator with a specific seed
 * @param seed Seed value
 */
void Random::setSeed(unsigned int seed) {
    eng.seed(seed);
}