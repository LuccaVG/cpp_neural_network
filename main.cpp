#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <fstream>
// Change the include path to use the local json.hpp file
#include "json.hpp"
#include "neural_network.h"
#include "chatbot.h"
#include "memory.h"

using json = nlohmann::json;

// Demo the neural network on the XOR problem
void runNeuralNetworkDemo() {
    std::cout << "\n===== Neural Network XOR Demo =====" << std::endl;
    
    // Define network architecture
    std::vector<int> layers = {2, 4, 1};
    
    // Create neural network with ReLU for hidden layer and sigmoid for output
    NeuralNetwork network(
        layers,
        NeuralNetwork::ActivationFunction::RELU,
        NeuralNetwork::ActivationFunction::SIGMOID
    );
    
    // Define training data (XOR problem)
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };
    
    // Set up training options
    NeuralNetwork::TrainingOptions options;
    options.epochs = 1000;
    options.learningRate = 0.01;
    options.optimizer = NeuralNetwork::Optimizer::ADAM;
    options.batchSize = 4;
    options.useL2Regularization = true;
    options.l2Lambda = 0.001;
    
    // Train network
    std::cout << "Training neural network..." << std::endl;
    network.train(inputs, targets, options);
    
    // Test network
    auto metrics = network.evaluateBinaryClassificationMetrics(inputs, targets);
    std::cout << "\nResults:" << std::endl;
    std::cout << "Accuracy: " << metrics["accuracy"] * 100 << "%" << std::endl;
    std::cout << "F1 Score: " << metrics["f1_score"] * 100 << "%" << std::endl;
    
    // Print predictions
    std::cout << "\nPredictions:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        auto output = network.predict(inputs[i]);
        std::cout << inputs[i][0] << " XOR " << inputs[i][1] 
                  << " = " << output[0] << " (expected " << targets[i][0] << ")" << std::endl;
    }
    
    // Print model summary
    std::cout << "\n" << network.getModelSummary() << std::endl;
}

void runSmartChatbotDemo() {
    std::cout << "\n===== Smart Chatbot Demo =====" << std::endl;
    
    // Create emotion classifier network
    std::vector<int> layers = {20, 16, 8, 5}; // 20 input features, 5 emotions
    NeuralNetwork emotionNN(
        layers,
        NeuralNetwork::ActivationFunction::RELU,
        NeuralNetwork::ActivationFunction::SOFTMAX
    );
    
    // Initialize memory system
    Memory memory(1000);
    
    // Initialize chatbot
    ChatBot chatbot(emotionNN, memory);
    
    std::string input;
    bool running = true;
    
    std::cout << "\nChatbot initialized. Type 'exit' to quit, or 'help' for commands." << std::endl;
    std::cout << "ChatBot: Hello! How can I help you today?" << std::endl;
    
    while (running) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        if (input == "exit" || input == "quit") {
            running = false;
            std::cout << "ChatBot: Goodbye!" << std::endl;
            memory.saveToFile("chatbot_memory.json");
        }
        else if (input == "help") {
            std::cout << "Available commands:\n"
                     << "  exit, quit - Exit the chatbot\n"
                     << "  help      - Display this help message\n"
                     << "  stats     - Show memory statistics\n"
                     << "  clear     - Clear conversation history\n";
        }
        else if (input == "stats") {
            std::cout << "\nMemory Statistics:\n";
            std::cout << "Total interactions: " << memory.size() << std::endl;
        }
        else if (input == "clear") {
            memory.clearMemory();
            std::cout << "ChatBot: Conversation history cleared." << std::endl;
        }
        else {
            // Process input and get response using the public generateResponse method
            auto response = chatbot.generateResponse(input);
            std::cout << "ChatBot: " << response << std::endl;
        }
    }
}

int main() {
    std::cout << "Neural Network and Smart Chatbot Demo" << std::endl;
    std::cout << "====================================" << std::endl;
    
    int choice;
    do {
        std::cout << "\nSelect an option:" << std::endl;
        std::cout << "1. Run Neural Network XOR Demo" << std::endl;
        std::cout << "2. Run Smart Chatbot" << std::endl;
        std::cout << "0. Exit" << std::endl;
        std::cout << "Choice: ";
        
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number." << std::endl;
            continue;
        }
        
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        
        switch (choice) {
            case 1:
                runNeuralNetworkDemo();
                break;
            case 2:
                runSmartChatbotDemo();
                break;
            case 0:
                std::cout << "Exiting program. Goodbye!" << std::endl;
                break;
            default:
                std::cout << "Invalid choice, please try again." << std::endl;
        }
    } while (choice != 0);
    
    return 0;
}