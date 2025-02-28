#include "memory.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
// Change the include path to use the local json.hpp file
#include "json.hpp"

using json = nlohmann::json;

// Memory Record implementation
MemoryRecord::MemoryRecord(
    const std::string& input,
    const std::string& response,
    const std::string& category,
    const std::string& sentiment,
    const std::vector<std::string>& keywords,
    double importance,
    time_t timestamp
) : input(input),
    response(response),
    category(category),
    sentiment(sentiment),
    keywords(keywords),
    importance(importance),
    timestamp(timestamp),
    accessCount(0),
    lastAccessTime(timestamp) {}

// Memory implementation
Memory::Memory(size_t capacity) : maxCapacity(capacity), isRunning(false) {
    // Initialize the task queue for background processing
    startBackgroundWorker();
}

Memory::~Memory() {
    // Stop the background worker thread
    stopBackgroundWorker();
}

// Helper methods defined at the top to avoid "undefined" errors
std::string Memory::normalizeText(const std::string& text) const {
    // Convert text to lowercase
    std::string normalized = text;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    return normalized;
}

void Memory::updateAccessMetadata(size_t index) const {
    // Update access count and time
    if (index < interactions.size()) {
        interactions[index].accessCount++;
        interactions[index].lastAccessTime = std::time(nullptr);
    }
}

double Memory::getRecencyScore(time_t timestamp) const {
    // Calculate recency score based on time difference
    // More recent timestamps get higher scores
    
    time_t currentTime = std::time(nullptr);
    double ageInHours = difftime(currentTime, timestamp) / (60 * 60);
    
    // Using sigmoid-like function to map age to [0,1] range
    // Very recent interactions (< 24h) get high scores
    // Older interactions get diminishing scores
    return 1.0 / (1.0 + ageInHours / 24.0);
}

double Memory::calculateSemanticSimilarity(
    const std::vector<std::string>& keywords1,
    const std::vector<std::string>& keywords2) const {
    
    if (keywords1.empty() || keywords2.empty()) {
        return 0.0;
    }
    
    // Count common keywords
    size_t commonCount = 0;
    for (const auto& keyword : keywords1) {
        if (std::find(keywords2.begin(), keywords2.end(), keyword) != keywords2.end()) {
            commonCount++;
        }
    }
    
    // Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
    size_t uniqueTotal = keywords1.size() + keywords2.size() - commonCount;
    if (uniqueTotal == 0) return 0.0;
    
    return static_cast<double>(commonCount) / uniqueTotal;
}

double Memory::calculateImportance(const std::string& input, 
                                  const std::string& category, 
                                  const std::string& sentiment) const {
    double importance = 0.5; // Base importance
    
    // Adjust based on input length (longer inputs might be more detailed/important)
    double lengthFactor = std::min(1.0, input.length() / 200.0);
    importance += lengthFactor * 0.1;
    
    // Adjust based on category
    if (category == "question" || category == "tech" || 
        category == "personal" || category == "help") {
        importance += 0.1;
    }
    
    // Adjust based on sentiment (strong emotions might indicate importance)
    if (sentiment == "very_positive" || sentiment == "very_negative") {
        importance += 0.2;
    } else if (sentiment == "positive" || sentiment == "negative") {
        importance += 0.1;
    }
    
    // Adjust based on question marks and exclamation marks
    size_t questionCount = std::count(input.begin(), input.end(), '?');
    size_t exclamationCount = std::count(input.begin(), input.end(), '!');
    importance += questionCount * 0.05;
    importance += exclamationCount * 0.03;
    
    // Cap importance at 1.0
    return std::min(1.0, importance);
}

// ADDED IMPLEMENTATIONS OF MISSING FUNCTIONS
void Memory::applyMemoryDecay() {
    // This function gradually reduces the importance of old memories
    // unless they have been accessed frequently
    
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    time_t currentTime = std::time(nullptr);
    
    for (auto& record : interactions) {
        // Calculate age of the memory in days
        double ageInDays = difftime(currentTime, record.timestamp) / (60 * 60 * 24);
        
        if (ageInDays < 7) {
            continue;  // Skip recent memories (less than a week old)
        }
        
        // Calculate decay factor based on age and access frequency
        // Older memories decay faster, but frequent access counteracts decay
        double ageFactor = std::min(1.0, ageInDays / 365.0); // Max effect after 1 year
        double accessFactor = std::max(0.0, 1.0 - std::log1p(record.accessCount) * 0.1);
        
        // Combined decay factor (0 to 0.2 range, so decay is gradual)
        double decayFactor = 0.2 * ageFactor * accessFactor;
        
        // Apply decay to importance (but don't go below 0.1)
        record.importance = std::max(0.1, record.importance * (1.0 - decayFactor));
    }
}

void Memory::startBackgroundWorker() {
    if (isRunning) {
        return; // Already running
    }
    
    isRunning = true;
    
    // Start the background thread
    backgroundThread = std::thread([this]() {
        while (isRunning) {
            // Wait for a task or timeout
            std::unique_lock<std::mutex> lock(taskMutex);
            bool hasTask = taskCondition.wait_for(lock, std::chrono::seconds(60),
                                              [this] { return !taskQueue.empty(); });
            
            if (hasTask) {
                // Get the next task
                auto task = std::move(taskQueue.front());
                taskQueue.pop();
                
                // Release the lock while executing the task
                lock.unlock();
                
                // Execute the task
                task();
                
                // Continue to next task
                continue;
            }
            
            // No task received within timeout, do periodic maintenance
            lock.unlock();
            applyMemoryDecay();
        }
    });
}

void Memory::stopBackgroundWorker() {
    if (!isRunning) {
        return; // Not running
    }
    
    {
        std::lock_guard<std::mutex> lock(taskMutex);
        isRunning = false;
    }
    
    // Wake up the worker thread
    taskCondition.notify_one();
    
    // Wait for the thread to finish
    if (backgroundThread.joinable()) {
        backgroundThread.join();
    }
    
    // Clear any remaining tasks
    std::queue<std::function<void()>> empty;
    std::swap(taskQueue, empty);
}

void Memory::enqueueBackgroundTask(std::function<void()> task) {
    if (!isRunning) {
        return; // Background worker not running
    }
    
    {
        std::lock_guard<std::mutex> lock(taskMutex);
        taskQueue.push(std::move(task));
    }
    
    // Notify worker thread
    taskCondition.notify_one();
}
// END OF ADDED IMPLEMENTATIONS

MemoryRecord Memory::mergeRecords(const MemoryRecord& record1, const MemoryRecord& record2) const {
    // Use the more recent input and response
    const MemoryRecord& primary = (record1.timestamp > record2.timestamp) ? record1 : record2;
    const MemoryRecord& secondary = (record1.timestamp > record2.timestamp) ? record2 : record1;
    
    // Merge keywords, removing duplicates
    std::vector<std::string> mergedKeywords = primary.keywords;
    for (const auto& keyword : secondary.keywords) {
        if (std::find(mergedKeywords.begin(), mergedKeywords.end(), keyword) == mergedKeywords.end()) {
            mergedKeywords.push_back(keyword);
        }
    }
    
    // Choose the higher importance
    double mergedImportance = std::max(primary.importance, secondary.importance);
    
    // Create the merged record
    MemoryRecord merged(
        primary.input,
        primary.response,
        primary.category,
        primary.sentiment,
        mergedKeywords,
        mergedImportance,
        primary.timestamp
    );
    
    // Set the merged access metadata
    merged.accessCount = record1.accessCount + record2.accessCount;
    merged.lastAccessTime = std::max(record1.lastAccessTime, record2.lastAccessTime);
    
    return merged;
}

void Memory::storeInteraction(
    const std::string& input,
    const std::string& response,
    const std::string& category,
    const std::string& sentiment
) {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Extract keywords for indexing and searching
    std::vector<std::string> keywords = extractKeywords(input);
    
    // Calculate importance based on various factors
    double importance = calculateImportance(input, category, sentiment);
    
    // Create new memory record
    MemoryRecord record(
        input,
        response,
        category,
        sentiment,
        keywords,
        importance,
        std::time(nullptr)
    );
    
    // Add record to interactions vector
    interactions.push_back(record);
    
    // Update indexes for fast retrieval
    updateIndexes(record, interactions.size() - 1);
    
    // Schedule background task to consolidate similar memories and apply decay
    enqueueBackgroundTask([this]() {
        consolidateMemories();
        applyMemoryDecay();
    });
    
    // Check if we've exceeded capacity
    if (interactions.size() > maxCapacity) {
        pruneMemories();
    }
}

std::vector<MemoryRecord> Memory::retrieveInteractions(size_t count) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    std::vector<MemoryRecord> result;
    size_t retrieveCount = std::min(count, interactions.size());
    
    // Reserve space for efficiency
    result.reserve(retrieveCount);
    
    // Copy the most recent interactions
    for (size_t i = 0; i < retrieveCount; ++i) {
        size_t index = interactions.size() - 1 - i;
        result.push_back(interactions[index]);
        
        // Update access metadata
        updateAccessMetadata(index);
    }
    
    return result;
}

std::vector<MemoryRecord> Memory::retrieveByKeyword(const std::string& keyword, size_t maxCount) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    std::vector<MemoryRecord> result;
    
    // Normalize the keyword to ensure case-insensitive matching
    std::string normalizedKeyword = normalizeText(keyword);
    
    // Check if we have this keyword in our index
    auto it = keywordIndex.find(normalizedKeyword);
    if (it != keywordIndex.end()) {
        // Create pairs of (index, relevance score) for sorting
        std::vector<std::pair<size_t, double>> indexedResults;
        
        for (const auto& idx : it->second) {
            // Calculate relevance for this record
            double relevance = calculateRelevanceScore(interactions[idx], normalizedKeyword);
            indexedResults.push_back({idx, relevance});
        }
        
        // Sort by relevance score
        std::sort(indexedResults.begin(), indexedResults.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Take only maxCount results
        size_t resultCount = std::min(maxCount, indexedResults.size());
        result.reserve(resultCount);
        
        for (size_t i = 0; i < resultCount; ++i) {
            size_t idx = indexedResults[i].first;
            result.push_back(interactions[idx]);
            
            // Update access metadata
            updateAccessMetadata(idx);
        }
    }
    
    return result;
}

std::vector<MemoryRecord> Memory::retrieveByCategory(const std::string& category, size_t maxCount) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    std::vector<MemoryRecord> result;
    
    // Normalize the category to ensure consistent matching
    std::string normalizedCategory = normalizeText(category);
    
    // Check if we have this category in our index
    auto it = categoryIndex.find(normalizedCategory);
    if (it != categoryIndex.end()) {
        // Create pairs of (index, importance) for sorting
        std::vector<std::pair<size_t, double>> indexedResults;
        
        for (const auto& idx : it->second) {
            double importanceScore = interactions[idx].importance;
            indexedResults.push_back({idx, importanceScore});
        }
        
        // Sort by importance score
        std::sort(indexedResults.begin(), indexedResults.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Take only maxCount results
        size_t resultCount = std::min(maxCount, indexedResults.size());
        result.reserve(resultCount);
        
        for (size_t i = 0; i < resultCount; ++i) {
            size_t idx = indexedResults[i].first;
            result.push_back(interactions[idx]);
            
            // Update access metadata
            updateAccessMetadata(idx);
        }
    }
    
    return result;
}

std::vector<MemoryRecord> Memory::retrieveBySentiment(const std::string& sentiment, size_t maxCount) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    std::vector<MemoryRecord> result;
    
    // Normalize the sentiment to ensure consistent matching
    std::string normalizedSentiment = normalizeText(sentiment);
    
    // Check if we have this sentiment in our index
    auto it = sentimentIndex.find(normalizedSentiment);
    if (it != sentimentIndex.end()) {
        // Create pairs of (index, recency) for sorting
        std::vector<std::pair<size_t, time_t>> indexedResults;
        
        for (const auto& idx : it->second) {
            time_t recency = interactions[idx].timestamp;
            indexedResults.push_back({idx, recency});
        }
        
        // Sort by recency (most recent first)
        std::sort(indexedResults.begin(), indexedResults.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Take only maxCount results
        size_t resultCount = std::min(maxCount, indexedResults.size());
        result.reserve(resultCount);
        
        for (size_t i = 0; i < resultCount; ++i) {
            size_t idx = indexedResults[i].first;
            result.push_back(interactions[idx]);
            
            // Update access metadata
            updateAccessMetadata(idx);
        }
    }
    
    return result;
}

std::vector<MemoryRecord> Memory::retrieveRecent(size_t count) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    std::vector<MemoryRecord> result;
    size_t retrieveCount = std::min(count, interactions.size());
    
    // Reserve space for efficiency
    result.reserve(retrieveCount);
    
    // Copy the most recent interactions (they're already stored chronologically)
    for (size_t i = 0; i < retrieveCount; ++i) {
        size_t index = interactions.size() - 1 - i;
        result.push_back(interactions[index]);
        
        // Update access metadata
        updateAccessMetadata(index);
    }
    
    return result;
}

std::vector<MemoryRecord> Memory::retrieveByImportance(size_t count) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Create a copy of all interactions for sorting
    std::vector<std::pair<size_t, double>> indexedResults;
    indexedResults.reserve(interactions.size());
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        indexedResults.push_back({i, interactions[i].importance});
    }
    
    // Sort by importance (highest first)
    std::sort(indexedResults.begin(), indexedResults.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take only 'count' results
    size_t resultCount = std::min(count, indexedResults.size());
    std::vector<MemoryRecord> result;
    result.reserve(resultCount);
    
    for (size_t i = 0; i < resultCount; ++i) {
        size_t idx = indexedResults[i].first;
        result.push_back(interactions[idx]);
        
        // Update access metadata
        updateAccessMetadata(idx);
    }
    
    return result;
}

std::vector<MemoryRecord> Memory::searchBySemanticSimilarity(const std::string& query, size_t count) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Extract keywords from the query
    std::vector<std::string> queryKeywords = extractKeywords(query);
    
    // Score all interactions based on semantic similarity to the query
    std::vector<std::pair<size_t, double>> scoredInteractions;
    scoredInteractions.reserve(interactions.size());
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        // Calculate semantic similarity between query and interaction
        double similarity = calculateSemanticSimilarity(queryKeywords, interactions[i].keywords);
        
        // Include importance and recency factors
        double adjustedScore = similarity * 0.6 + 
                              interactions[i].importance * 0.3 + 
                              getRecencyScore(interactions[i].timestamp) * 0.1;
        
        scoredInteractions.push_back({i, adjustedScore});
    }
    
    // Sort by score (highest first)
    std::sort(scoredInteractions.begin(), scoredInteractions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take only the top 'count' results
    size_t resultCount = std::min(count, scoredInteractions.size());
    std::vector<MemoryRecord> result;
    result.reserve(resultCount);
    
    for (size_t i = 0; i < resultCount; ++i) {
        size_t idx = scoredInteractions[i].first;
        result.push_back(interactions[idx]);
        
        // Update access metadata
        updateAccessMetadata(idx);
    }
    
    return result;
}

void Memory::clearMemory() {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    interactions.clear();
    keywordIndex.clear();
    categoryIndex.clear();
    sentimentIndex.clear();
}

void Memory::pruneOldestRecords(size_t keepCount) {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    if (keepCount >= interactions.size()) {
        return; // Nothing to prune
    }
    
    // Calculate how many records to remove
    size_t removeCount = interactions.size() - keepCount;
    
    // Remove oldest records (which are at the beginning of the vector)
    interactions.erase(interactions.begin(), interactions.begin() + removeCount);
    
    // Rebuild indexes since vector positions have changed
    rebuildIndexes();
}

size_t Memory::size() const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    return interactions.size();
}

bool Memory::isEmpty() const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    return interactions.empty();
}

bool Memory::saveToFile(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    try {
        // Create JSON array for all interactions
        json memoryJson = json::array();
        
        for (const auto& record : interactions) {
            json recordJson = {
                {"input", record.input},
                {"response", record.response},
                {"category", record.category},
                {"sentiment", record.sentiment},
                {"keywords", record.keywords},
                {"importance", record.importance},
                {"timestamp", record.timestamp},
                {"access_count", record.accessCount},
                {"last_access", record.lastAccessTime}
            };
            
            memoryJson.push_back(recordJson);
        }
        
        // Write to file with pretty formatting
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << std::setw(4) << memoryJson << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving memory to file: " << e.what() << std::endl;
        return false;
    }
}

bool Memory::loadFromFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    try {
        // Open and read the file
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        json memoryJson;
        file >> memoryJson;
        
        // Clear existing interactions
        interactions.clear();
        keywordIndex.clear();
        categoryIndex.clear();
        sentimentIndex.clear();
        
        // Parse JSON and create memory records
        for (const auto& recordJson : memoryJson) {
            MemoryRecord record(
                recordJson["input"],
                recordJson["response"],
                recordJson["category"],
                recordJson["sentiment"],
                recordJson["keywords"].get<std::vector<std::string>>(),
                recordJson["importance"],
                recordJson["timestamp"]
            );
            
            // Restore access metadata if available
            if (recordJson.contains("access_count")) {
                record.accessCount = recordJson["access_count"];
            }
            if (recordJson.contains("last_access")) {
                record.lastAccessTime = recordJson["last_access"];
            }
            
            interactions.push_back(record);
        }
        
        // Rebuild all indexes
        rebuildIndexes();
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading memory from file: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::pair<std::string, int>> Memory::getTopKeywords(size_t count) const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Create map to count keyword frequencies
    std::unordered_map<std::string, int> keywordCounts;
    
    // Count occurrences of each keyword across all interactions
    for (const auto& record : interactions) {
        for (const auto& keyword : record.keywords) {
            keywordCounts[keyword]++;
        }
    }
    
    // Convert map to vector for sorting
    std::vector<std::pair<std::string, int>> keywordFrequencies;
    keywordFrequencies.reserve(keywordCounts.size());
    
    for (const auto& pair : keywordCounts) {
        keywordFrequencies.push_back(pair);
    }
    
    // Sort by frequency (highest first)
    std::sort(keywordFrequencies.begin(), keywordFrequencies.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return only the top 'count' keywords
    if (keywordFrequencies.size() > count) {
        keywordFrequencies.resize(count);
    }
    
    return keywordFrequencies;
}

std::vector<std::pair<std::string, int>> Memory::getCategoryCounts() const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Create map to count category frequencies
    std::unordered_map<std::string, int> categoryCounts;
    
    // Count occurrences of each category
    for (const auto& record : interactions) {
        categoryCounts[record.category]++;
    }
    
    // Convert map to vector for sorting
    std::vector<std::pair<std::string, int>> categoryFrequencies;
    categoryFrequencies.reserve(categoryCounts.size());
    
    for (const auto& pair : categoryCounts) {
        categoryFrequencies.push_back(pair);
    }
    
    // Sort by frequency (highest first)
    std::sort(categoryFrequencies.begin(), categoryFrequencies.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return categoryFrequencies;
}

std::vector<std::pair<std::string, int>> Memory::getSentimentDistribution() const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Create map to count sentiment frequencies
    std::unordered_map<std::string, int> sentimentCounts;
    
    // Count occurrences of each sentiment
    for (const auto& record : interactions) {
        sentimentCounts[record.sentiment]++;
    }
    
    // Convert map to vector for sorting
    std::vector<std::pair<std::string, int>> sentimentFrequencies;
    sentimentFrequencies.reserve(sentimentCounts.size());
    
    for (const auto& pair : sentimentCounts) {
        sentimentFrequencies.push_back(pair);
    }
    
    // Sort by frequency (highest first)
    std::sort(sentimentFrequencies.begin(), sentimentFrequencies.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return sentimentFrequencies;
}

std::vector<std::string> Memory::extractKeywords(const std::string& text) const {
    // Stop words to exclude
    static const std::unordered_set<std::string> stopWords = {
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "can", "could", "may", "might", "must",
        "to", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "dont", "should", "now"
    };
    
    std::vector<std::string> keywords;
    
    // Convert to lowercase and tokenize
    std::string lowercaseText = normalizeText(text);
    std::istringstream iss(lowercaseText);
    std::string word;
    
    // Process each word
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), 
                                 [](char c) { return std::ispunct(c); }), word.end());
        
        // Skip empty words, short words, and stop words
        if (word.empty() || word.length() < 3 || stopWords.find(word) != stopWords.end()) {
            continue;
        }
        
        // Add to keywords
        keywords.push_back(word);
    }
    
    return keywords;
}

void Memory::updateIndexes(const MemoryRecord& record, size_t position) {
    // Update keyword index
    for (const auto& keyword : record.keywords) {
        keywordIndex[keyword].push_back(position);
    }
    
    // Update category index
    categoryIndex[normalizeText(record.category)].push_back(position);
    
    // Update sentiment index
    sentimentIndex[normalizeText(record.sentiment)].push_back(position);
}

void Memory::rebuildIndexes() {
    // Clear existing indexes
    keywordIndex.clear();
    categoryIndex.clear();
    sentimentIndex.clear();
    
    // Rebuild indexes for all interactions
    for (size_t i = 0; i < interactions.size(); ++i) {
        updateIndexes(interactions[i], i);
    }
}

double Memory::calculateRelevanceScore(const MemoryRecord& record, const std::string& keyword) const {
    // Base relevance score starts with 1.0
    double score = 1.0;
    
    // Keyword match in the record keywords boosts score
    bool keywordFound = false;
    for (const auto& k : record.keywords) {
        if (normalizeText(k) == normalizeText(keyword)) {
            score *= 1.5;
            keywordFound = true;
            break;
        }
    }
    
    // If no direct keyword match, lower the score
    if (!keywordFound) {
        score *= 0.5;
    }
    
    // Importance factor: more important memories are more relevant
    score *= (0.5 + record.importance);
    
    // Recency factor: more recent memories are slightly more relevant
    score *= (0.8 + 0.2 * getRecencyScore(record.timestamp));
    
    // Access frequency factor: frequently accessed memories are more relevant
    score *= (0.9 + std::min(0.1, 0.01 * record.accessCount));
    
    return score;
}
void Memory::consolidateMemories() {
    // This is a complex operation that looks for similar memories and merges them
    // It's executed in the background to avoid blocking the main thread
    
    if (interactions.size() < 10) {
        return; // Not enough memories to consolidate
    }
    
    // Find similar memory pairs using semantic similarity
    std::vector<std::pair<size_t, size_t>> similarPairs;
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        for (size_t j = i + 1; j < interactions.size(); ++j) {
            // Calculate semantic similarity between this pair of memories
            double similarity = calculateSemanticSimilarity(
                interactions[i].keywords,
                interactions[j].keywords
            );
            
            // If similarity exceeds threshold, consider merging
            if (similarity > 0.6) {  // Using 0.6 as threshold for "similar enough"
                similarPairs.push_back({i, j});
            }
        }
    }
    
    if (similarPairs.empty()) {
        return;  // No similar memories found
    }
    
    // Process pairs in order of similarity (highest first)
    std::sort(similarPairs.begin(), similarPairs.end(),
              [this](const auto& a, const auto& b) {
                  double simA = calculateSemanticSimilarity(
                      interactions[a.first].keywords,
                      interactions[a.second].keywords
                  );
                  
                  double simB = calculateSemanticSimilarity(
                      interactions[b.first].keywords,
                      interactions[b.second].keywords
                  );
                  
                  return simA > simB;
              });
    
    // Keep track of which memories have been merged
    std::unordered_set<size_t> mergedIndexes;
    
    // Create a new vector for consolidated memories
    std::vector<MemoryRecord> consolidatedMemories;
    consolidatedMemories.reserve(interactions.size());
    
    // Process each pair
    for (const auto& pair : similarPairs) {
        size_t i = pair.first;
        size_t j = pair.second;
        
        // Skip if either memory has already been merged
        if (mergedIndexes.find(i) != mergedIndexes.end() ||
            mergedIndexes.find(j) != mergedIndexes.end()) {
            continue;
        }
        
        // Merge these memories
        MemoryRecord merged = mergeRecords(interactions[i], interactions[j]);
        
        // Add to consolidated memories
        consolidatedMemories.push_back(merged);
        
        // Mark as merged
        mergedIndexes.insert(i);
        mergedIndexes.insert(j);
    }
    
    // Add all non-merged memories
    for (size_t i = 0; i < interactions.size(); ++i) {
        if (mergedIndexes.find(i) == mergedIndexes.end()) {
            consolidatedMemories.push_back(interactions[i]);
        }
    }
    
    // Replace the original interactions with consolidated memories
    interactions = std::move(consolidatedMemories);
    
    // Rebuild indexes to reflect changes
    rebuildIndexes();
}

void Memory::pruneMemories() {
    // This function removes the least important memories when capacity is exceeded
    
    // If we haven't exceeded capacity, nothing to do
    if (interactions.size() <= maxCapacity) {
        return;
    }
    
    // Calculate how many memories to remove
    size_t removeCount = interactions.size() - maxCapacity;
    
    // Create a vector of pairs: (index, score) where score determines which to keep
    std::vector<std::pair<size_t, double>> scoredMemories;
    scoredMemories.reserve(interactions.size());
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        // Score is a combination of importance, recency, and access count
        double score = interactions[i].importance * 0.5 +
                       getRecencyScore(interactions[i].timestamp) * 0.3 + 
                       std::min(1.0, interactions[i].accessCount / 10.0) * 0.2;
                       
        scoredMemories.push_back({i, score});
    }
    
    // Sort by score (lowest first, as we'll remove these)
    std::sort(scoredMemories.begin(), scoredMemories.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Collect indexes to remove (the lowest scoring ones)
    std::vector<size_t> indexesToRemove;
    indexesToRemove.reserve(removeCount);
    
    for (size_t i = 0; i < removeCount; ++i) {
        indexesToRemove.push_back(scoredMemories[i].first);
    }
    
    // Sort indexes in descending order to avoid shifting issues when removing
    std::sort(indexesToRemove.begin(), indexesToRemove.end(), std::greater<size_t>());
    
    // Remove memories starting from the highest index
    for (size_t idx : indexesToRemove) {
        interactions.erase(interactions.begin() + idx);
    }
    
    // Rebuild indexes to reflect the changes
    rebuildIndexes();
}