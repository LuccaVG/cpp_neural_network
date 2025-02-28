#include "chatbot.h"
#include <iostream>
#include <sstream>

ChatBot::ChatBot(NeuralNetwork& nn, Memory& memory) : nn(nn), memory(memory) {}

void ChatBot::chat() {
    std::string input;
    std::vector<double> inputs(3);

    std::cout << "ChatBot: Hello! How can I assist you today?" << std::endl;

    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, input);

        if (input == "exit") {
            std::cout << "ChatBot: Goodbye!" << std::endl;
            break;
        }

        // Convert input string to vector of doubles (for simplicity, assume 3 inputs)
        std::istringstream iss(input);
        for (double& value : inputs) {
            iss >> value;
        }

        std::string response = generateResponse(inputs);
        std::cout << "ChatBot: " << response << std::endl;
    }
}

std::string ChatBot::generateResponse(const std::vector<double>& inputs) {
    std::vector<double> outputs = nn.feedForward(inputs);
    std::ostringstream oss;
    oss << "Processed inputs: ";
    for (double output : outputs) {
        oss << output << " ";
    }
    return oss.str();
}