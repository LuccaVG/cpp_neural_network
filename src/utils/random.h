#ifndef RANDOM_H
#define RANDOM_H

/**
 * @brief Utility class for random number generation
 */
class Random {
public:
    /**
     * @brief Generate a random double in the range [min, max].
     * @param min Minimum value.
     * @param max Maximum value.
     * @return Random double value.
     */
    static double generateDouble(double min, double max);

    /**
     * @brief Generate a random integer in the range [min, max].
     * @param min Minimum value.
     * @param max Maximum value.
     * @return Random integer value.
     */
    static int generateInt(int min, int max);

    /**
     * @brief Generate random values with normal distribution
     * @param mean Mean of the normal distribution
     * @param stddev Standard deviation of the normal distribution
     * @return Random value from normal distribution
     */
    static double generateNormal(double mean = 0.0, double stddev = 1.0);

    /**
     * @brief Initialize the random number generator with a specific seed
     * @param seed Seed value
     */
    static void setSeed(unsigned int seed);
};

#endif // RANDOM_H