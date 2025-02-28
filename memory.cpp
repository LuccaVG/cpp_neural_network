#include "memory.h"

bool Memory::exists(int id) const {
    return memory.find(id) != memory.end();
}

void Memory::store(const std::vector<double>& value) {
    memory[counter++] = value;
}

std::vector<double> Memory::retrieve(int id) const {
    auto it = memory.find(id);
    if (it != memory.end()) {
        return it->second;
    }
    return {};
}