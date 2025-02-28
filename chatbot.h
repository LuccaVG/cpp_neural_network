#ifndef CHATBOT_H
#define CHATBOT_H

#include <string>
#include <vector>
#include "neural_network.h"
#include "memory.h"

class ChatBot {
public:
    ChatBot(NeuralNetwork& nn, Memory& memory);
    void chat();

private:
    NeuralNetwork& nn;
    Memory& memory;
    std::string generateResponse(const std::vector<double>& inputs);
};

#endif // CHATBOT_H