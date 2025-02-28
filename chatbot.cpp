#include "chatbot.h"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <curl/curl.h>
#include <regex>
#include <random>
#include <ctime>
#include <thread>
#include <future>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Helper struct for CURL operations
struct CurlBuffer {
    char* data;
    size_t size;
    
    CurlBuffer() : data(nullptr), size(0) {}
    
    ~CurlBuffer() {
        if (data) free(data);
    }
};

// CURL write callback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    CurlBuffer* mem = static_cast<CurlBuffer*>(userp);
    
    char* ptr = static_cast<char*>(realloc(mem->data, mem->size + realsize + 1));
    if (!ptr) {
        // Out of memory
        return 0;
    }
    
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    
    return realsize;
}

ChatBot::ChatBot(NeuralNetwork& nn, Memory& memory) 
    : nn(nn), memory(memory), sentimentAnalyzer(SentimentAnalyzer()), isRunning(false) {
    
    // Initialize CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Initialize categories and responses
    initializeCategories();
    initializeResponses();
    
    // Initialize conversation context
    conversationContext.clear();
}

ChatBot::~ChatBot() {
    curl_global_cleanup();
}

void ChatBot::chat() {
    std::string input;
    isRunning = true;
    
    std::cout << "Bot: Hello! I'm an AI assistant. How can I help you today?" << std::endl;
    
    while (isRunning) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        if (input == "exit" || input == "quit" || input == "bye") {
            std::cout << "Bot: Goodbye! Have a great day!" << std::endl;
            isRunning = false;
            break;
        }
        
        // Process input in a separate thread for responsiveness
        auto future = std::async(std::launch::async, &ChatBot::processInputAsync, this, input);
        
        // Show thinking animation
        showThinkingAnimation(future);
        
        // Get response
        std::string response = future.get();
        std::cout << "Bot: " << response << std::endl;
    }
}

std::string ChatBot::processInputAsync(const std::string& input) {
    // Store interaction in memory
    std::string processedInput = processInput(input);
    
    // Analyze sentiment and intent
    std::string sentiment = sentimentAnalyzer.analyzeSentiment(processedInput);
    std::string category = categorizeInput(processedInput);
    
    // Generate response
    std::string response = generateResponse(processedInput, sentiment, category);
    
    // Store the full interaction with metadata
    memory.storeInteraction(input, response, category, sentiment);
    
    // Update conversation context
    updateConversationContext(input, response, category);
    
    return response;
}

void ChatBot::showThinkingAnimation(std::future<std::string>& future) {
    std::string animation = "|/-\\";
    int animationIdx = 0;
    
    while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
        std::cout << "\rThinking " << animation[animationIdx % animation.length()] << std::flush;
        animationIdx++;
    }
    
    std::cout << "\r          \r" << std::flush;
}

std::string ChatBot::processInput(const std::string& input) {
    // Convert input to lowercase
    std::string processedInput = input;
    std::transform(processedInput.begin(), processedInput.end(), processedInput.begin(), ::tolower);
    
    // Remove excessive punctuation but keep sentence structure
    processedInput = std::regex_replace(processedInput, std::regex("\\s+"), " ");
    processedInput = std::regex_replace(processedInput, std::regex("([.!?])\\1+"), "$1");
    
    return processedInput;
}

std::string ChatBot::generateResponse(const std::string& input, const std::string& sentiment, const std::string& category) {
    // Check if we have a direct match in our predefined responses
    for (const auto& pattern : responsePatterns) {
        std::regex re(pattern.first, std::regex::icase);
        if (std::regex_search(input, re)) {
            return getRandomResponse(pattern.second);
        }
    }
    
    // Check if we need to fetch web content
    if (shouldFetchWebContent(input)) {
        std::string webContent = fetchWebContent(input);
        if (!webContent.empty()) {
            return "Based on web information: " + webContent;
        }
    }
    
    // Check if we have similar past interactions
    std::vector<MemoryRecord> similarInteractions = memory.retrieveByKeyword(extractMainKeyword(input), 3);
    if (!similarInteractions.empty()) {
        // Use the most relevant past response
        return "I remember discussing this before. " + similarInteractions[0].response;
    }
    
    // Generate a response based on sentiment and category
    if (!categoryResponses[category].empty()) {
        std::string baseResponse = getRandomResponse(categoryResponses[category]);
        
        // Modify based on sentiment
        if (sentiment == "positive") {
            return baseResponse + " I'm glad you're feeling positive about this!";
        } else if (sentiment == "negative") {
            return baseResponse + " I understand this might be challenging for you.";
        } else {
            return baseResponse;
        }
    }
    
    // Fallback response
    return "I'm still learning about " + category + ". Can you tell me more?";
}

void ChatBot::initializeCategories() {
    // Define regex patterns for different categories
    categoryPatterns = {
        {"greeting", "\\b(hi|hello|hey|good (morning|afternoon|evening))\\b"},
        {"farewell", "\\b(bye|goodbye|see you|talk to you later)\\b"},
        {"question", "\\b(what|who|where|when|why|how)\\b.+\\?"},
        {"weather", "\\b(weather|temperature|forecast|rain|snow|sunny|cloudy)\\b"},
        {"help", "\\b(help|assist|support|guide)\\b"},
        {"tech", "\\b(computer|software|hardware|program|code|app|technology)\\b"},
        {"personal", "\\b(feel|think|believe|opinion|view)\\b"}
    };
}

void ChatBot::initializeResponses() {
    // Define response patterns (regex -> possible responses)
    responsePatterns = {
        {"\\b(hi|hello|hey|greetings)\\b", {
            "Hello there! How can I assist you today?",
            "Hi! What can I help you with?",
            "Greetings! How may I be of service?"
        }},
        {"\\bhow are you\\b", {
            "I'm functioning well, thank you for asking! How about you?",
            "I'm doing great! How can I help you today?",
            "All systems operational! What's on your mind?"
        }},
        {"\\bthank(s| you)\\b", {
            "You're welcome! Is there anything else I can help with?",
            "Happy to assist! Let me know if you need anything else.",
            "My pleasure! What else would you like to know?"
        }}
    };
    
    // Category-specific responses
    categoryResponses = {
        {"greeting", {
            "Hello! How can I assist you today?",
            "Hi there! What brings you here?",
            "Greetings! How may I help you?"
        }},
        {"farewell", {
            "Goodbye! Have a great day!",
            "Take care! Feel free to come back anytime.",
            "See you later! It was nice chatting with you."
        }},
        {"question", {
            "That's a good question. Let me think about it.",
            "I'll try my best to answer that for you.",
            "Interesting question! Here's what I know:"
        }},
        {"weather", {
            "Would you like me to check the weather forecast for you?",
            "Weather patterns can be quite interesting to analyze.",
            "Let me see if I can find the latest weather information for you."
        }},
        {"help", {
            "I'm here to help! What specifically do you need assistance with?",
            "I'd be happy to assist you. Could you provide more details?",
            "How can I best support you with this?"
        }},
        {"tech", {
            "Technology is advancing rapidly these days.",
            "I find tech topics fascinating. What aspect are you interested in?",
            "I'm designed to help with various technical questions."
        }},
        {"personal", {
            "I appreciate you sharing your thoughts.",
            "That's an interesting perspective.",
            "I understand how you feel about this."
        }},
        {"general", {
            "Tell me more about that.",
            "That's interesting. Could you elaborate?",
            "I'd like to understand better what you mean."
        }}
    };
}

std::string ChatBot::categorizeInput(const std::string& input) {
    for (const auto& category : categoryPatterns) {
        std::regex pattern(category.second, std::regex::icase);
        if (std::regex_search(input, pattern)) {
            return category.first;
        }
    }
    return "general"; // Default category
}

std::string ChatBot::getRandomResponse(const std::vector<std::string>& responses) {
    if (responses.empty()) return "I'm not sure what to say.";
    
    static std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<std::size_t> distribution(0, responses.size() - 1);
    
    return responses[distribution(rng)];
}

bool ChatBot::shouldFetchWebContent(const std::string& input) {
    // Keywords that might trigger web search
    std::vector<std::string> webSearchTriggers = {
        "search", "look up", "find information", "what is", "tell me about",
        "latest news", "current events", "weather in"
    };
    
    for (const auto& trigger : webSearchTriggers) {
        if (input.find(trigger) != std::string::npos)#include "chatbot.h"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <curl/curl.h>
#include <regex>
#include <random>
#include <ctime>
#include <thread>
#include <future>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Helper struct for CURL operations
struct CurlBuffer {
    char* data;
    size_t size;
    
    CurlBuffer() : data(nullptr), size(0) {}
    
    ~CurlBuffer() {
        if (data) free(data);
    }
};

// CURL write callback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    CurlBuffer* mem = static_cast<CurlBuffer*>(userp);
    
    char* ptr = static_cast<char*>(realloc(mem->data, mem->size + realsize + 1));
    if (!ptr) {
        // Out of memory
        return 0;
    }
    
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    
    return realsize;
}

ChatBot::ChatBot(NeuralNetwork& nn, Memory& memory) 
    : nn(nn), memory(memory), sentimentAnalyzer(SentimentAnalyzer()), isRunning(false) {
    
    // Initialize CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Initialize categories and responses
    initializeCategories();
    initializeResponses();
    
    // Initialize conversation context
    conversationContext.clear();
}

ChatBot::~ChatBot() {
    curl_global_cleanup();
}

void ChatBot::chat() {
    std::string input;
    isRunning = true;
    
    std::cout << "Bot: Hello! I'm an AI assistant. How can I help you today?" << std::endl;
    
    while (isRunning) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        if (input == "exit" || input == "quit" || input == "bye") {
            std::cout << "Bot: Goodbye! Have a great day!" << std::endl;
            isRunning = false;
            break;
        }
        
        // Process input in a separate thread for responsiveness
        auto future = std::async(std::launch::async, &ChatBot::processInputAsync, this, input);
        
        // Show thinking animation
        showThinkingAnimation(future);
        
        // Get response
        std::string response = future.get();
        std::cout << "Bot: " << response << std::endl;
    }
}

std::string ChatBot::processInputAsync(const std::string& input) {
    // Store interaction in memory
    std::string processedInput = processInput(input);
    
    // Analyze sentiment and intent
    std::string sentiment = sentimentAnalyzer.analyzeSentiment(processedInput);
    std::string category = categorizeInput(processedInput);
    
    // Generate response
    std::string response = generateResponse(processedInput, sentiment, category);
    
    // Store the full interaction with metadata
    memory.storeInteraction(input, response, category, sentiment);
    
    // Update conversation context
    updateConversationContext(input, response, category);
    
    return response;
}

void ChatBot::showThinkingAnimation(std::future<std::string>& future) {
    std::string animation = "|/-\\";
    int animationIdx = 0;
    
    while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
        std::cout << "\rThinking " << animation[animationIdx % animation.length()] << std::flush;
        animationIdx++;
    }
    
    std::cout << "\r          \r" << std::flush;
}

std::string ChatBot::processInput(const std::string& input) {
    // Convert input to lowercase
    std::string processedInput = input;
    std::transform(processedInput.begin(), processedInput.end(), processedInput.begin(), ::tolower);
    
    // Remove excessive punctuation but keep sentence structure
    processedInput = std::regex_replace(processedInput, std::regex("\\s+"), " ");
    processedInput = std::regex_replace(processedInput, std::regex("([.!?])\\1+"), "$1");
    
    return processedInput;
}

std::string ChatBot::generateResponse(const std::string& input, const std::string& sentiment, const std::string& category) {
    // Check if we have a direct match in our predefined responses
    for (const auto& pattern : responsePatterns) {
        std::regex re(pattern.first, std::regex::icase);
        if (std::regex_search(input, re)) {
            return getRandomResponse(pattern.second);
        }
    }
    
    // Check if we need to fetch web content
    if (shouldFetchWebContent(input)) {
        std::string webContent = fetchWebContent(input);
        if (!webContent.empty()) {
            return "Based on web information: " + webContent;
        }
    }
    
    // Check if we have similar past interactions
    std::vector<MemoryRecord> similarInteractions = memory.retrieveByKeyword(extractMainKeyword(input), 3);
    if (!similarInteractions.empty()) {
        // Use the most relevant past response
        return "I remember discussing this before. " + similarInteractions[0].response;
    }
    
    // Generate a response based on sentiment and category
    if (!categoryResponses[category].empty()) {
        std::string baseResponse = getRandomResponse(categoryResponses[category]);
        
        // Modify based on sentiment
        if (sentiment == "positive") {
            return baseResponse + " I'm glad you're feeling positive about this!";
        } else if (sentiment == "negative") {
            return baseResponse + " I understand this might be challenging for you.";
        } else {
            return baseResponse;
        }
    }
    
    // Fallback response
    return "I'm still learning about " + category + ". Can you tell me more?";
}

void ChatBot::initializeCategories() {
    // Define regex patterns for different categories
    categoryPatterns = {
        {"greeting", "\\b(hi|hello|hey|good (morning|afternoon|evening))\\b"},
        {"farewell", "\\b(bye|goodbye|see you|talk to you later)\\b"},
        {"question", "\\b(what|who|where|when|why|how)\\b.+\\?"},
        {"weather", "\\b(weather|temperature|forecast|rain|snow|sunny|cloudy)\\b"},
        {"help", "\\b(help|assist|support|guide)\\b"},
        {"tech", "\\b(computer|software|hardware|program|code|app|technology)\\b"},
        {"personal", "\\b(feel|think|believe|opinion|view)\\b"}
    };
}

void ChatBot::initializeResponses() {
    // Define response patterns (regex -> possible responses)
    responsePatterns = {
        {"\\b(hi|hello|hey|greetings)\\b", {
            "Hello there! How can I assist you today?",
            "Hi! What can I help you with?",
            "Greetings! How may I be of service?"
        }},
        {"\\bhow are you\\b", {
            "I'm functioning well, thank you for asking! How about you?",
            "I'm doing great! How can I help you today?",
            "All systems operational! What's on your mind?"
        }},
        {"\\bthank(s| you)\\b", {
            "You're welcome! Is there anything else I can help with?",
            "Happy to assist! Let me know if you need anything else.",
            "My pleasure! What else would you like to know?"
        }}
    };
    
    // Category-specific responses
    categoryResponses = {
        {"greeting", {
            "Hello! How can I assist you today?",
            "Hi there! What brings you here?",
            "Greetings! How may I help you?"
        }},
        {"farewell", {
            "Goodbye! Have a great day!",
            "Take care! Feel free to come back anytime.",
            "See you later! It was nice chatting with you."
        }},
        {"question", {
            "That's a good question. Let me think about it.",
            "I'll try my best to answer that for you.",
            "Interesting question! Here's what I know:"
        }},
        {"weather", {
            "Would you like me to check the weather forecast for you?",
            "Weather patterns can be quite interesting to analyze.",
            "Let me see if I can find the latest weather information for you."
        }},
        {"help", {
            "I'm here to help! What specifically do you need assistance with?",
            "I'd be happy to assist you. Could you provide more details?",
            "How can I best support you with this?"
        }},
        {"tech", {
            "Technology is advancing rapidly these days.",
            "I find tech topics fascinating. What aspect are you interested in?",
            "I'm designed to help with various technical questions."
        }},
        {"personal", {
            "I appreciate you sharing your thoughts.",
            "That's an interesting perspective.",
            "I understand how you feel about this."
        }},
        {"general", {
            "Tell me more about that.",
            "That's interesting. Could you elaborate?",
            "I'd like to understand better what you mean."
        }}
    };
}

std::string ChatBot::categorizeInput(const std::string& input) {
    for (const auto& category : categoryPatterns) {
        std::regex pattern(category.second, std::regex::icase);
        if (std::regex_search(input, pattern)) {
            return category.first;
        }
    }
    return "general"; // Default category
}

std::string ChatBot::getRandomResponse(const std::vector<std::string>& responses) {
    if (responses.empty()) return "I'm not sure what to say.";
    
    static std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<std::size_t> distribution(0, responses.size() - 1);
    
    return responses[distribution(rng)];
}

bool ChatBot::shouldFetchWebContent(const std::string& input) {
    // Keywords that might trigger web search
    std::vector<std::string> webSearchTriggers = {
        "search", "look up", "find information", "what is", "tell me about",
        "latest news", "current events", "weather in"
    };
    
    for (const auto& trigger : webSearchTriggers) {
        if (input.find(trigger) != std::string::npos) {
            return true;
        }
    }
    
    // Check if input looks like a factual question
    std::regex factualQuestionPattern("\\b(what|who|where|when|why|how) (is|are|was|were|do|does|did) \\b", std::regex::icase);
    if (std::regex_search(input, factualQuestionPattern)) {
        return true;
    }
    
    return false;
}

std::string ChatBot::fetchWebContent(const std::string& query) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return "Sorry, I couldn't connect to external resources.";
    }
    
    // Extract search terms
    std::string searchQuery = extractSearchQuery(query);
    if (searchQuery.empty()) {
        return "I couldn't determine what to search for.";
    }
    
    // Encode search query for URL
    char* encodedQuery = curl_easy_escape(curl, searchQuery.c_str(), static_cast<int>(searchQuery.length()));
    if (!encodedQuery) {
        curl_easy_cleanup(curl);
        return "Sorry, I couldn't process the search query.";
    }
    
    // Form API request URL (using DuckDuckGo API as an example)
    std::string url = "https://api.duckduckgo.com/?q=" + std::string(encodedQuery) + "&format=json";
    curl_free(encodedQuery);
    
    // Set up CURL request
    CurlBuffer buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "ChatBot/1.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10);
    
    // Execute request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        return "Sorry, I couldn't retrieve information from the web right now.";
    }
    
    curl_easy_cleanup(curl);
    
    // Parse response
    try {
        if (buffer.data && buffer.size > 0) {
            json response = json::parse(buffer.data);
            
            // Extract abstract text if available
            if (response.contains("AbstractText") && !response["AbstractText"].empty()) {
                return response["AbstractText"];
            }
            
            // If no abstract, try to find other useful information
            if (response.contains("RelatedTopics") && response["RelatedTopics"].is_array() && 
                !response["RelatedTopics"].empty() && 
                response["RelatedTopics"][0].contains("Text")) {
                return response["RelatedTopics"][0]["Text"];
            }
        }
    } catch (const std::exception& e) {
        return "I found some information but couldn't process it correctly.";
    }
    
    return "I couldn't find relevant information about that.";
}

std::string ChatBot::extractMainKeyword(const std::string& input) {
    // Remove common stop words
    std::vector<std::string> stopWords = {
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "can", "could", "may", "might", "must",
        "to", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "don't", "should", "now"
    };
    
    // Tokenize input
    std::istringstream iss(input);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        // Convert to lowercase and remove punctuation
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        token.erase(std::remove_if(token.begin(), token.end(), ::ispunct), token.end());
        
        // Skip empty tokens and stop words
        if (token.empty() || std::find(stopWords.begin(), stopWords.end(), token) != stopWords.end()) {
            continue;
        }
        
        tokens.push_back(token);
    }
    
    if (tokens.empty()) {
        return "";
    }
    
    // Find the most frequent non-stop word
    std::unordered_map<std::string, int> wordCounts;
    for (const auto& word : tokens) {
        wordCounts[word]++;
    }
    
    std::string mainKeyword = "";
    int maxCount = 0;
    for (const auto& pair : wordCounts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            mainKeyword = pair.first;
        }
    }
    
    return mainKeyword;
}

std::string ChatBot::extractSearchQuery(const std::string& input) {
    // Common patterns to extract search queries
    std::vector<std::pair<std::regex, int>> extractionPatterns = {
        {std::regex("search for (.*)", std::regex::icase), 1},
        {std::regex("look up (.*)", std::regex::icase), 1},
        {std::regex("find information about (.*)", std::regex::icase), 1},
        {std::regex("what is (.*?)\\??$", std::regex::icase), 1},
        {std::regex("tell me about (.*)", std::regex::icase), 1},
        {std::regex("who is (.*?)\\??$", std::regex::icase), 1},
        {std::regex("where is (.*?)\\??$", std::regex::icase), 1},
        {std::regex("when is (.*?)\\??$", std::regex::icase), 1},
        {std::regex("how does (.*) work\\??$", std::regex::icase), 1}
    };
    
    std::smatch matches;
    for (const auto& pattern : extractionPatterns) {
        if (std::regex_search(input, matches, pattern.first) && matches.size() > pattern.second) {
            return matches[pattern.second].str();
        }
    }
    
    // If no patterns matched, extract keywords after removing question words and stop words
    std::string processed = std::regex_replace(input, std::regex("\\b(what|who|where|when|why|how|is|are|do|does|did|can|could|would|should|tell me about|search for)\\b", std::regex::icase), "");
    processed = std::regex_replace(processed, std::regex("\\?"), "");
    processed = std::regex_replace(processed, std::regex("^\\s+|\\s+$"), "");
    
    return processed;
}

void ChatBot::updateConversationContext(const std::string& input, const std::string& response, const std::string& category) {
    // Store recent interactions for context
    const size_t MAX_CONTEXT_SIZE = 10;
    
    conversationContext.push_back({
        {"input", input},
        {"response", response},
        {"category", category},
        {"timestamp", std::to_string(std::time(nullptr))}
    });
    
    // Maintain limited context size
    if (conversationContext.size() > MAX_CONTEXT_SIZE) {
        conversationContext.erase(conversationContext.begin());
    }
    
    // Update topic tracking
    std::vector<std::string> keywords = extractKeywords(input);
    for (const auto& keyword : keywords) {
        topicFrequency[keyword]++;
    }
}

std::vector<std::string> ChatBot::extractKeywords(const std::string& text) {
    // Similar to extractMainKeyword but returns multiple keywords
    std::vector<std::string> stopWords = {
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "can", "could", "may", "might", "must",
        "to", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "don't", "should", "now"
    };
    
    // Tokenize input
    std::istringstream iss(text);
    std::vector<std::string> keywords;
    std::string token;
    
    while (iss >> token) {
        // Convert to lowercase and remove punctuation
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        token.erase(std::remove_if(token.begin(), token.end(), ::ispunct), token.end());
        
        // Skip empty tokens and stop words
        if (token.empty() || token.size() < 3 || 
            std::find(stopWords.begin(), stopWords.end(), token) != stopWords.end()) {
            continue;
        }
        
        keywords.push_back(token);
    }
    
    return keywords;
}

std::string ChatBot::getCurrentTopic() const {
    if (topicFrequency.empty()) {
        return "general";
    }
    
    // Find the most discussed topic
    std::string currentTopic = "";
    int maxFrequency = 0;
    
    for (const auto& pair : topicFrequency) {
        if (pair.second > maxFrequency) {
            maxFrequency = pair.second;
            currentTopic = pair.first;
        }
    }
    
    return currentTopic;
}

void ChatBot::resetConversation() {
    conversationContext.clear();
    topicFrequency.clear();
}

// Enhanced SentimentAnalyzer implementation
SentimentAnalyzer::SentimentAnalyzer() {
    // Initialize positive and negative word lists
    positiveWords = {
        "good", "great", "excellent", "wonderful", "fantastic", "amazing", "awesome",
        "happy", "joy", "love", "like", "pleased", "delighted", "satisfied", "positive",
        "brilliant", "extraordinary", "spectacular", "outstanding", "superb", "perfect",
        "incredible", "remarkable", "beautiful", "impressive", "exceptional", "terrific",
        "marvelous", "splendid", "enjoyable", "favorable", "pleasant", "fine", "decent",
        "nice", "super", "fun", "glad", "grateful", "thankful", "appreciate", "excited",
        "enthusiastic", "eager", "interested", "hopeful", "optimistic", "confident"
    };
    
    negativeWords = {
        "bad", "terrible", "horrible", "awful", "dreadful", "poor", "disappointing",
        "sad", "unhappy", "hate", "dislike", "angry", "upset", "frustrated", "negative",
        "terrible", "appalling", "atrocious", "dire", "disgusting", "foul", "nasty",
        "offensive", "unpleasant", "abysmal", "deficient", "inadequate", "inferior",
        "unsatisfactory", "abominable", "adverse", "alarming", "annoying", "awkward",
        "boring", "broken", "confused", "corrupt", "crazy", "creepy", "cruel", "dangerous",
        "depressing", "dirty", "dishonest", "embarrassing", "evil", "failed", "frightening",
        "guilty", "harmful", "harsh", "hurt", "illegal", "insane", "insecure", "jealous",
        "lonely", "mean", "messy", "miserable", "mistake", "painful", "pathetic", "problem",
        "rejected", "rotten", "rude", "scared", "severe", "shame", "shocking", "sloppy",
        "sorry", "stupid", "suspicious", "tense", "terrible", "tired", "tough", "tragic",
        "ugly", "unattractive", "uncomfortable", "unfortunate", "unhealthy", "unlucky",
        "unpleasant", "unsatisfactory", "unwanted", "unwelcome", "upset", "useless",
        "weak", "worried", "worthless", "wrong"
    };
    
    // Add context-sensitive sentiment phrases
    sentimentPhrases = {
        {"not good", -1},
        {"not bad", 1},
        {"not great", -1},
        {"very good", 2},
        {"very bad", -2},
        {"really like", 2},
        {"don't like", -2},
        {"really hate", -3},
        {"absolutely love", 3},
        {"can't stand", -2},
        {"really disappointed", -2},
        {"extremely happy", 3}
    };
}

std::string SentimentAnalyzer::analyzeSentiment(const std::string& text) {
    int sentimentScore = calculateSentimentScore(text);
    
    if (sentimentScore > 2) {
        return "very_positive";
    } else if (sentimentScore > 0) {
        return "positive";
    } else if (sentimentScore == 0) {
        return "neutral";
    } else if (sentimentScore > -3) {
        return "negative";
    } else {
        return "very_negative";
    }
}

int SentimentAnalyzer::calculateSentimentScore(const std::string& text) {
    std::string lowerText = text;
    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
    
    int score = 0;
    
    // Check for sentiment phrases first (context-sensitive)
    for (const auto& phrase : sentimentPhrases) {
        if (lowerText.find(phrase.first) != std::string::npos) {
            score += phrase.second;
        }
    }
    
    // Check for individual words
    std::istringstream iss(lowerText);
    std::string word;
    bool negationActive = false;
    
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        
        // Handle negation (like "not good")
        if (word == "not" || word == "don't" || word == "doesn't" || word == "didn't" || 
            word == "isn't" || word == "aren't" || word == "wasn't" || word == "weren't" ||
            word == "won't" || word == "wouldn't" || word == "shouldn't" || word == "can't" ||
            word == "cannot" || word == "couldn't" || word == "never") {
            negationActive = true;
            continue;
        }
        
        // Check word against positive and negative lists
        if (std::find(positiveWords.begin(), positiveWords.end(), word) != positiveWords.end()) {
            score += negationActive ? -1 : 1;
        } else if (std::find(negativeWords.begin(), negativeWords.end(), word) != negativeWords.end()) {
            score += negationActive ? 1 : -1;
        }
        
        // Reset negation after using it or after punctuation
        if (negationActive && (word.find('.') != std::string::npos || word.find('!') != std::string::npos || 
            word.find('?') != std::string::npos || word.find(',') != std::string::npos)) {
            negationActive = false;
        }
    }
    
    return score;
}