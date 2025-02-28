#include <iostream>
#include "neural_network.h"
#include "memory.h"
#include "chatbot.h"
#include <vector>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Enhanced neural network with more layers and neurons
    NeuralNetwork nn({3, 10, 10, 5, 1});
    
    // Improved memory management
    Memory memory;
    
    // Advanced chatbot with NLP capabilities
    ChatBot chatbot(nn, memory);

    std::cout << "Starting advanced AI system...\n";
    chatbot.chat();

    return 0;
}