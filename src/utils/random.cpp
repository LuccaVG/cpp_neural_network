#include <random>
#include "random.h"

/**
 * @brief Generate a random double in the range [min, max].
 * @param min Minimum value.
 * @param max Maximum value.
 * @return Random double value.
 */
double Random::generateDouble(double min, double max) {
    static std::random_device rd;  // Obtain a random number from hardware
    static std::mt19937 eng(rd());  // Seed the generator
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
    static std::random_device rd;  // Obtain a random number from hardware
    static std::mt19937 eng(rd());  // Seed the generator
    std::uniform_int_distribution<> distr(min, max);  // Define the range
    return distr(eng);
}