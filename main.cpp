#include <iostream>
#include "neural_network.h"
#include "memory.h"
#include "chatbot.h"
#include <vector>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    NeuralNetwork nn({3, 5, 3, 1});
    Memory memory;
    ChatBot chatbot(nn, memory);

    std::cout << "Starting AI system...\n";
    chatbot.chat();

    return 0;
}