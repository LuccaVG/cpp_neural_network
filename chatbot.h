#ifndef CHATBOT_H
#define CHATBOT_H

#include "neural_network.h"
#include "memory.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <future>
#include <sstream>
// Change the include path to use the local json.hpp file
#include "json.hpp"

// Curl buffer structure for web requests
struct CurlBuffer {
    char* data;
    size_t size;
    
    CurlBuffer() : data(nullptr), size(0) {}
    ~CurlBuffer() { if (data) free(data); }
};

// Callback function for curl
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    CurlBuffer* buffer = static_cast<CurlBuffer*>(userp);
    
    char* ptr = static_cast<char*>(realloc(buffer->data, buffer->size + realsize + 1));
    if (!ptr) {
        return 0;  // Out of memory
    }
    
    buffer->data = ptr;
    memcpy(&(buffer->data[buffer->size]), contents, realsize);
    buffer->size += realsize;
    buffer->data[buffer->size] = 0;
    
    return realsize;
}

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
    void resetConversation();
    
private:
    NeuralNetwork& nn;
    Memory& memory;
    SentimentAnalyzer sentimentAnalyzer;
    bool isRunning;
    std::regex factualQuestionPattern;
    std::vector<std::string> webSearchTriggers;
    
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

     // Time-related helper methods
     std::string getCurrentTimeResponse() const;
     std::string getCurrentDateResponse() const;  
     std::string getCurrentDateTimeResponse() const;
    
    // Initialization methods
    void initializeCategories();
    void initializeResponses();
};

#endif // CHATBOT_H