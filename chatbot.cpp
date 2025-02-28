#include "chatbot.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cctype>
#include <regex>
#include <thread>
#include <chrono>
#include <iomanip> // Required for std::put_time

// SentimentAnalyzer implementation
SentimentAnalyzer::SentimentAnalyzer() {
    // Initialize positive words
    positiveWords = {
        "good", "great", "excellent", "wonderful", "fantastic",
        "happy", "love", "enjoy", "awesome", "amazing",
        "positive", "helpful", "useful", "beautiful", "best",
        "better", "like", "appreciate", "thanks", "gratitude"
    };
    
    // Initialize negative words
    negativeWords = {
        "bad", "terrible", "awful", "horrible", "poor",
        "sad", "hate", "dislike", "unhappy", "worse",
        "worst", "useless", "ugly", "negative", "wrong",
        "difficult", "problem", "issue", "error", "sorry"
    };
    
    // Initialize sentiment phrases with scores
    sentimentPhrases = {
        {"really good", 2}, {"very bad", -2},
        {"extremely happy", 3}, {"very sad", -3},
        {"not good", -1}, {"not bad", 1},
        {"absolutely love", 3}, {"totally hate", -3},
        {"highly recommend", 2}, {"would avoid", -2}
    };
}

std::string SentimentAnalyzer::analyzeSentiment(const std::string& text) {
    int score = calculateSentimentScore(text);
    
    if (score > 5) return "very_positive";
    if (score > 1) return "positive";
    if (score < -5) return "very_negative";
    if (score < -1) return "negative";
    return "neutral";
}

int SentimentAnalyzer::calculateSentimentScore(const std::string& text) {
    // Convert input to lowercase for case-insensitive matching
    std::string lowercase = text;
    std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);
    
    int score = 0;
    
    // Check for sentiment phrases
    for (const auto& phrase : sentimentPhrases) {
        if (lowercase.find(phrase.first) != std::string::npos) {
            score += phrase.second;
        }
    }
    
    // Check for positive words
    for (const auto& word : positiveWords) {
        std::regex wordRegex("\\b" + word + "\\b");
        std::sregex_iterator it(lowercase.begin(), lowercase.end(), wordRegex);
        std::sregex_iterator end;
        while (it != end) {
            score++;
            ++it;
        }
    }
    
    // Check for negative words
    for (const auto& word : negativeWords) {
        std::regex wordRegex("\\b" + word + "\\b");
        std::sregex_iterator it(lowercase.begin(), lowercase.end(), wordRegex);
        std::sregex_iterator end;
        while (it != end) {
            score--;
            ++it;
        }
    }
    
    return score;
}

// ChatBot implementation
ChatBot::ChatBot(NeuralNetwork& nn, Memory& memory)
    : nn(nn), memory(memory), isRunning(true) {
    
    // Initialize patterns
    initializeCategories();
    initializeResponses();
    
    // Set up question pattern regex
    factualQuestionPattern = std::regex("\\b(what|who|how|when|where|why)\\b.+\\?");
    
    // Initialize search triggers
    webSearchTriggers = {"search", "find", "lookup", "google", "information", "look up"};
}

ChatBot::~ChatBot() {
    isRunning = false;
}

void ChatBot::chat() {
    std::cout << "ChatBot initialized. Type 'exit' to quit." << std::endl;
    std::string input;
    
    while (isRunning) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        if (input == "exit") {
            isRunning = false;
            break;
        }
        
        std::string response = generateResponse(input);
        std::cout << "ChatBot: " << response << std::endl;
    }
    
    std::cout << "ChatBot shutting down. Goodbye!" << std::endl;
}

// Time-related functions
std::string ChatBot::getCurrentTimeResponse() const {
    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    
    // Format time nicely in HH:MM:SS format
    std::stringstream ss;
    ss << "The current time is " << std::put_time(std::localtime(&currentTime), "%I:%M:%S %p");
    return ss.str();
}

std::string ChatBot::getCurrentDateResponse() const {
    // Get current date
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    
    // Format date nicely in Month Day, Year format
    std::stringstream ss;
    ss << "Today is " << std::put_time(std::localtime(&currentTime), "%B %d, %Y");
    return ss.str();
}

std::string ChatBot::getCurrentDateTimeResponse() const {
    // Get current date and time
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    
    // Format date and time nicely
    std::stringstream ss;
    ss << "It is " << std::put_time(std::localtime(&currentTime), "%B %d, %Y at %I:%M:%S %p");
    return ss.str();
}

std::string ChatBot::generateResponse(const std::string& input, 
                                     const std::string& sentiment,
                                     const std::string& category) {
    // Categorize input if category not provided
    std::string determinedCategory = category == "general" ? categorizeInput(input) : category;
    
    // Analyze sentiment if not provided
    std::string determinedSentiment = sentiment == "neutral" ? 
                                     sentimentAnalyzer.analyzeSentiment(input) : sentiment;
    
    // Process input to generate response
    std::string response = processInput(input);
    
    // Store the interaction in memory
    memory.storeInteraction(input, response, determinedCategory, determinedSentiment);
    
    // Update conversation context
    updateConversationContext(input, response, determinedCategory);
    
    return response;
}

void ChatBot::resetConversation() {
    conversationContext.clear();
    topicFrequency.clear();
}

std::string ChatBot::processInput(const std::string& input) {
    // Convert input to lowercase for easier matching
    std::string lowerInput = input;
    std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);
    
    // Check for time/date related queries
    if (lowerInput.find("time") != std::string::npos && 
        (lowerInput.find("what") != std::string::npos || lowerInput.find("tell") != std::string::npos)) {
        return getCurrentTimeResponse();
    }
    
    if (lowerInput.find("date") != std::string::npos && 
        (lowerInput.find("what") != std::string::npos || lowerInput.find("tell") != std::string::npos)) {
        return getCurrentDateResponse();
    }
    
    if ((lowerInput.find("day") != std::string::npos || lowerInput.find("today") != std::string::npos) &&
        (lowerInput.find("what") != std::string::npos || lowerInput.find("tell") != std::string::npos)) {
        return getCurrentDateResponse();
    }
    
    // Check for combined date and time query
    if ((lowerInput.find("date") != std::string::npos && lowerInput.find("time") != std::string::npos) ||
        lowerInput.find("now") != std::string::npos) {
        return getCurrentDateTimeResponse();
    }
    
    // Try to find direct response patterns
    for (const auto& pattern : responsePatterns) {
        std::regex patternRegex(pattern.first, std::regex::icase);
        if (std::regex_search(input, patternRegex)) {
            return getRandomResponse(pattern.second);
        }
    }
    
    // Get category-specific responses
    std::string category = categorizeInput(input);
    
    // Check if this is a factual question
    if (std::regex_search(input, factualQuestionPattern)) {
        if (shouldFetchWebContent(input)) {
            std::string query = extractSearchQuery(input);
            return fetchWebContent(query);
        }
    }
    
    // Use memory to find contextually relevant response
    auto relevantMemories = memory.searchBySemanticSimilarity(input, 3);
    if (!relevantMemories.empty()) {
        // Check if we have a highly similar previous interaction
        double similarityThreshold = 0.8;
        
        // For simplicity, we'll use the most relevant memory
        return relevantMemories[0].response;
    }
    
    // Get response based on category if available
    auto it = categoryResponses.find(category);
    if (it != categoryResponses.end()) {
        return getRandomResponse(it->second);
    }
    
    // Fallback to neural network response
    // (This is a placeholder - actual NN integration would be more complex)
    std::vector<double> inputVector;
    // Convert input to vector representation...
    std::vector<double> outputVector = nn.predict(inputVector);
    
    // Convert output vector to text response...
    // For now, just use a default response
    return "I'm processing that information. Can you tell me more?";
}

std::string ChatBot::processInputAsync(const std::string& input) {
    // Simulate some processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(500 + rand() % 1000));
    return processInput(input);
}

void ChatBot::showThinkingAnimation(std::future<std::string>& future) {
    const std::vector<std::string> frames = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
    int i = 0;
    
    while (future.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready) {
        std::cout << "\rThinking " << frames[i % frames.size()] << std::flush;
        i++;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "\r          \r" << std::flush;
}

std::string ChatBot::categorizeInput(const std::string& input) {
    // Check each category pattern
    for (const auto& pattern : categoryPatterns) {
        std::regex patternRegex(pattern.second, std::regex::icase);
        if (std::regex_search(input, patternRegex)) {
            return pattern.first;
        }
    }
    
    // Check for questions
    if (std::regex_search(input, factualQuestionPattern)) {
        return "question";
    }
    
    // Default category
    return "general";
}

std::string ChatBot::getRandomResponse(const std::vector<std::string>& responses) {
    if (responses.empty()) {
        return "I'm not sure how to respond to that.";
    }
    
    int index = rand() % responses.size();
    return responses[index];
}

bool ChatBot::shouldFetchWebContent(const std::string& input) {
    // Check if input contains any search triggers
    std::string lowercaseInput = input;
    std::transform(lowercaseInput.begin(), lowercaseInput.end(), lowercaseInput.begin(), ::tolower);
    
    for (const auto& trigger : webSearchTriggers) {
        if (lowercaseInput.find(trigger) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

std::string ChatBot::fetchWebContent(const std::string& query) {
    // This is a placeholder - in a real implementation, we'd use CURL to fetch web content
    // For now, just indicate what we would search for
    return "I would search for information about: " + query;
}

std::string ChatBot::extractMainKeyword(const std::string& input) {
    // Extract keywords
    auto keywords = extractKeywords(input);
    if (!keywords.empty()) {
        // Return the most frequent keyword in conversation history
        std::string bestKeyword = keywords[0];
        int highestFreq = 0;
        
        for (const auto& keyword : keywords) {
            if (topicFrequency[keyword] > highestFreq) {
                highestFreq = topicFrequency[keyword];
                bestKeyword = keyword;
            }
        }
        
        return bestKeyword;
    }
    
    return "";
}

std::string ChatBot::extractSearchQuery(const std::string& input) {
    // Remove question words and punctuation
    std::string query = input;
    std::regex questionWords("\\b(what|who|how|when|where|why|is|are|do|does|can|could|would|will|should)\\b", 
                           std::regex::icase);
    query = std::regex_replace(query, questionWords, "");
    
    // Remove question marks and extra spaces
    query = std::regex_replace(query, std::regex("\\?"), "");
    query = std::regex_replace(query, std::regex("\\s+"), " ");
    query = std::regex_replace(query, std::regex("^\\s+|\\s+$"), "");
    
    return query.empty() ? input : query;
}

std::vector<std::string> ChatBot::extractKeywords(const std::string& text) {
    // This is a simplified keyword extraction
    // In a real implementation, we would use more sophisticated NLP techniques
    
    std::vector<std::string> stopWords = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "is", "are", "am", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "shall", "should", "may", "might",
        "can", "could", "i", "you", "he", "she", "it", "we", "they", "this", "that",
        "these", "those", "my", "your", "his", "her", "its", "our", "their"
    };
    
    std::vector<std::string> keywords;
    std::string word;
    std::istringstream iss(text);
    
    while (iss >> word) {
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), 
                                 [](char c) { return std::ispunct(c); }), word.end());
        
        // Check if word is long enough and not a stop word
        if (word.length() > 2 && 
            std::find(stopWords.begin(), stopWords.end(), word) == stopWords.end()) {
            keywords.push_back(word);
            topicFrequency[word]++;
        }
    }
    
    return keywords;
}

void ChatBot::updateConversationContext(const std::string& input, 
                                       const std::string& response, 
                                       const std::string& category) {
    // Create a JSON object for this interaction
    nlohmann::json interaction;
    interaction["input"] = input;
    interaction["response"] = response;
    interaction["category"] = category;
    interaction["timestamp"] = std::time(nullptr);
    
    // Add to context
    conversationContext.push_back(interaction);
    
    // Limit context size to prevent memory bloat
    const size_t MAX_CONTEXT_SIZE = 20;
    if (conversationContext.size() > MAX_CONTEXT_SIZE) {
        conversationContext.erase(conversationContext.begin());
    }
    
    // Update keywords frequency
    for (const auto& keyword : extractKeywords(input)) {
        topicFrequency[keyword]++;
    }
}

std::string ChatBot::getCurrentTopic() const {
    if (topicFrequency.empty()) {
        return "";
    }
    
    // Find most frequent topic
    auto it = std::max_element(
        topicFrequency.begin(), topicFrequency.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );
    
    return it->first;
}

void ChatBot::initializeCategories() {
    // Add regex patterns for different categories
    categoryPatterns["greeting"] = "\\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\\b";
    categoryPatterns["farewell"] = "\\b(goodbye|bye|see you|talk to you later)\\b";
    categoryPatterns["thanks"] = "\\b(thank you|thanks|appreciate it)\\b";
    categoryPatterns["help"] = "\\b(help|assist|guide|support)\\b";
    categoryPatterns["opinion"] = "\\b(think|opinion|feel about|view on)\\b";
    categoryPatterns["personal"] = "\\b(you|your name|about you)\\b";
    categoryPatterns["emotional"] = "\\b(sad|happy|angry|excited|worried|anxious|love|hate)\\b";
    categoryPatterns["tech"] = "\\b(computer|software|hardware|programming|code|technology|app|website)\\b";
}

void ChatBot::initializeResponses() {
    // Add response patterns
    responsePatterns["\\b(hello|hi|hey)\\b"] = {
        "Hello! How are you today?",
        "Hi there! Nice to meet you.",
        "Hey! How can I help you today?"
    };
    
    responsePatterns["\\b(how are you)\\b"] = {
        "I'm doing well, thank you for asking!",
        "I'm good! How about you?",
        "I'm functioning optimally, thanks for asking."
    };
    
    responsePatterns["\\b(thank you|thanks)\\b"] = {
        "You're welcome!",
        "Glad I could help!",
        "My pleasure!"
    };
    
    responsePatterns["\\b(goodbye|bye)\\b"] = {
        "Goodbye! Have a great day!",
        "Bye for now!",
        "See you later!"
    };
    
    responsePatterns["\\bname\\b"] = {
        "My name is ChatBot. I'm an AI assistant.",
        "I'm ChatBot, your friendly AI companion!",
        "You can call me ChatBot."
    };
    
    // Add category-specific responses
    categoryResponses["greeting"] = {
        "Hello there! How can I assist you today?",
        "Hi! It's great to see you!",
        "Hello! How are you doing today?"
    };
    
    categoryResponses["farewell"] = {
        "Goodbye! Have a wonderful day!",
        "Bye for now! Come back soon!",
        "Take care! Looking forward to our next conversation!"
    };
    
    categoryResponses["thanks"] = {
        "You're welcome! It's my pleasure to help.",
        "Glad I could assist you!",
        "Anytime! Feel free to ask if you need anything else."
    };
    
    categoryResponses["help"] = {
        "I'd be happy to help! What do you need assistance with?",
        "Sure, I can help with that. Could you provide more details?",
        "I'm here to help. What information are you looking for?"
    };
    
    categoryResponses["opinion"] = {
        "That's an interesting topic! I think there are multiple perspectives to consider.",
        "I'm designed to provide information rather than opinions, but I can explain different viewpoints.",
        "I try to present balanced information rather than personal opinions."
    };
    
    categoryResponses["personal"] = {
        "I'm ChatBot, an AI assistant designed to be helpful and informative.",
        "I'm a neural network-based chatbot created to have conversations and provide information.",
        "I'm an AI companion focused on providing helpful responses to your questions."
    };
    
    categoryResponses["emotional"] = {
        "I understand emotions are important. How can I support you right now?",
        "I'm here to listen if you want to talk about how you're feeling.",
        "Emotions are an important part of the human experience. Would you like to discuss yours?"
    };
    
    categoryResponses["tech"] = {
        "Technology is a fascinating field! What aspect are you interested in?",
        "I enjoy discussing technology. What would you like to know more about?",
        "Tech topics are among my favorites. What specific area are you curious about?"
    };
    
    categoryResponses["question"] = {
        "That's a good question. Let me think about that...",
        "Interesting question! Here's what I know...",
        "Let me provide you with the information I have on that topic..."
    };
    
    categoryResponses["general"] = {
        "That's interesting. Tell me more about that.",
        "I see. Would you like to elaborate?",
        "Interesting perspective. How did you come to that conclusion?"
    };
}