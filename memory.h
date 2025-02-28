#ifndef MEMORY_H
#define MEMORY_H

#include <unordered_map>
#include <vector>

class Memory {
public:
    void store(const std::vector<double>& value);
    std::vector<double> retrieve(int id) const;
    bool exists(int id) const;

private:
    std::unordered_map<int, std::vector<double>> memory;
    int counter = 0;
};

#endif // MEMORY_H