#ifndef CHATBOT_H
#define CHATBOT_H

#include "neural_network.h"
#include "memory.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <future>
#include <nlohmann/json.hpp>

// Forward declaration
class Memory;
class NeuralNetwork;

// Sentiment analysis component
class SentimentAnalyzer {
public:
    SentimentAnalyzer();
    std::string analyzeSentiment(const std::string& text);
    
private:
    std::vector<std::string> positiveWords;
    std::vector<std::string> negativeWords;
    std::unordered_map<std::string, int> sentimentPhrases;
    
    int calculateSentimentScore(const std::string& text);
};

// Main chatbot class
class ChatBot {
public:
    ChatBot(NeuralNetwork& nn, Memory& memory);
    ~ChatBot();
    
    void chat();
    std::string generateResponse(const std::string& input, const std::string& sentiment = "neutral", 
                                const std::string& category = "general");
    
private:
    NeuralNetwork& nn;
    Memory& memory;
    SentimentAnalyzer sentimentAnalyzer;
    bool isRunning;
    
    // Pattern recognition
    std::unordered_map<std::string, std::string> categoryPatterns;
    std::unordered_map<std::string, std::vector<std::string>> responsePatterns;
    std::unordered_map<std::string, std::vector<std::string>> categoryResponses;
    
    // Conversation context
    std::vector<nlohmann::json> conversationContext;
    std::unordered_map<std::string, int> topicFrequency;
    
    // Helper methods
    std::string processInput(const std::string& input);
    std::string processInputAsync(const std::string& input);
    void showThinkingAnimation(std::future<std::string>& future);
    std::string categorizeInput(const std::string& input);
    std::string getRandomResponse(const std::vector<std::string>& responses);
    bool shouldFetchWebContent(const std::string& input);
    std::string fetchWebContent(const std::string& query);
    std::string extractMainKeyword(const std::string& input);
    std::string extractSearchQuery(const std::string& input);
    std::vector<std::string> extractKeywords(const std::string& text);
    void updateConversationContext(const std::string& input, const std::string& response, const std::string& category);
    std::string getCurrentTopic() const;
    void resetConversation();
    
    // Initialization methods
    void initializeCategories();
    void initializeResponses();
};

#endif // CHATBOT_H