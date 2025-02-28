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
#include <nlohmann/json.hpp>

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
Memory::Memory(size_t capacity) : maxCapacity(capacity) {
    // Initialize the task queue for background processing
    startBackgroundWorker();
}

Memory::~Memory() {
    // Stop the background worker thread
    stopBackgroundWorker();
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
    // Base relevance score starts with 1.0 if the keyword is in the record's keywords
    double score = 0.0;
    
    // Check if the keyword is in the record's keywords
    if (std::find(record.keywords.begin(), record.keywords.end(), keyword) != record.keywords.end()) {
        score = 1.0;
    }
    
    // Factor in importance
    score *= (0.5 + record.importance * 0.5);
    
    // Factor in recency (more recent interactions get higher scores)
    score *= getRecencyScore(record.timestamp);
    
    // Factor in access frequency
    score *= (1.0 + std::log1p(record.accessCount) * 0.1);
    
    return score;
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

void Memory::pruneMemories() {
    if (interactions.size() <= maxCapacity) {
        return;
    }
    
    // Create a vector of (index, score) pairs for scoring each memory
    std::vector<std::pair<size_t, double>> scoredMemories;
    scoredMemories.reserve(interactions.size());
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        // Calculate retention score based on importance, recency, and access patterns
        double importanceScore = interactions[i].importance;
        double recencyScore = getRecencyScore(interactions[i].timestamp);
        double accessScore = std::log1p(interactions[i].accessCount) * 0.2;
        
        // Calculate final retention score
        double retentionScore = importanceScore * 0.5 + recencyScore * 0.3 + accessScore * 0.2;
        
        scoredMemories.push_back({i, retentionScore});
    }
    
    // Sort by retention score (lowest first)
    std::sort(scoredMemories.begin(), scoredMemories.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Calculate how many memories to remove to get to maxCapacity
    size_t removeCount = interactions.size() - maxCapacity;
    
    // Collect indexes to remove (converting to set for O(1) lookup)
    std::unordered_set<size_t> indexesToRemove;
    for (size_t i = 0; i < removeCount; ++i) {
        indexesToRemove.insert(scoredMemories[i].first);
    }
    
    // Create new vector without the removed memories
    std::vector<MemoryRecord> newInteractions;
    newInteractions.reserve(maxCapacity);
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        if (indexesToRemove.find(i) == indexesToRemove.end()) {
            newInteractions.push_back(interactions[i]);
        }
    }
    
    // Replace interactions with filtered vector
    interactions = std::move(newInteractions);
    
    // Rebuild indexes to reflect changes
    rebuildIndexes();
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
            double similarity = calculateSemanticSimilarity(
                interactions[i].keywords, interactions[j].keywords);
            
            // If similarity is high, consider them for consolidation
            if (similarity > 0.7) {
                similarPairs.push_back({i, j});
            }
        }
    }
    
    // Process similar pairs (from newest to oldest)
    std::sort(similarPairs.begin(), similarPairs.end(),
              [this](const auto& a, const auto& b) {
                  return std::max(interactions[a.first].timestamp, interactions[a.second].timestamp) >
                         std::max(interactions[b.first].timestamp, interactions[b.second].timestamp);
              });
    
    // Set to track which memories have been consolidated
    std::unordered_set<size_t> consolidated;
    
    // New vector for consolidated memories
    std::vector<MemoryRecord> newInteractions;
    newInteractions.reserve(interactions.size());
    
    // Process similar pairs
    for (const auto& pair : similarPairs) {
        size_t i = pair.first;
        size_t j = pair.second;
        
        // Skip if either memory has already been consolidated
        if (consolidated.find(i) != consolidated.end() || 
            consolidated.find(j) != consolidated.end()) {
            continue;
        }
        
        // Mark both memories as consolidated
        consolidated.insert(i);
        consolidated.insert(j);
        
        // Merge the two memories
        MemoryRecord mergedRecord = mergeRecords(interactions[i], interactions[j]);
        newInteractions.push_back(mergedRecord);
    }
    
    // Add remaining memories that weren't consolidated
    for (size_t i = 0; i < interactions.size(); ++i) {
        if (consolidated.find(i) == consolidated.end()) {
            newInteractions.push_back(interactions[i]);
        }
    }
    
    // Replace with consolidated memories
    std::lock_guard<std::mutex> lock(memoryMutex);
    interactions = std::move(newInteractions);
    rebuildIndexes();
}

MemoryRecord Memory::mergeRecords(const MemoryRecord& record1, const MemoryRecord& record2) const {
    // Create a new record that merges information from both source records
    
    // Use the more recent input and response
    const MemoryRecord& primary = (record1.timestamp > record2.timestamp) ? record1 : record2;
    const MemoryRecord& secondary = (record1.timestamp > record2.timestamp) ? record2 : record1;
    
    // Combine keywords
    std::vector<std::string> mergedKeywords = record1.keywords;
    for (const auto& keyword : record2.keywords) {
        if (std::find(mergedKeywords.begin(), mergedKeywords.end(), keyword) == mergedKeywords.end()) {
            mergedKeywords.push_back(keyword);
        }
    }
    
    // Use max importance
    double mergedImportance = std::max(record1.importance, record2.importance);
    
    // Use most recent timestamp
    time_t mergedTimestamp = std::max(record1.timestamp, record2.timestamp);
    
    // Sum access counts
    int mergedAccessCount = record1.accessCount + record2.accessCount;
    
    // Use most recent access time
    time_t mergedLastAccess = std::max(record1.lastAccessTime, record2.lastAccessTime);
    
    // Create merged record
    MemoryRecord mergedRecord(
        primary.input,
        primary.response,
        primary.category,
        primary.sentiment,
        mergedKeywords,
        mergedImportance,#include "memory.h"
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
#include <nlohmann/json.hpp>

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
Memory::Memory(size_t capacity) : maxCapacity(capacity) {
    // Initialize the task queue for background processing
    startBackgroundWorker();
}

Memory::~Memory() {
    // Stop the background worker thread
    stopBackgroundWorker();
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
    // Base relevance score starts with 1.0 if the keyword is in the record's keywords
    double score = 0.0;
    
    // Check if the keyword is in the record's keywords
    if (std::find(record.keywords.begin(), record.keywords.end(), keyword) != record.keywords.end()) {
        score = 1.0;
    }
    
    // Factor in importance
    score *= (0.5 + record.importance * 0.5);
    
    // Factor in recency (more recent interactions get higher scores)
    score *= getRecencyScore(record.timestamp);
    
    // Factor in access frequency
    score *= (1.0 + std::log1p(record.accessCount) * 0.1);
    
    return score;
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

void Memory::pruneMemories() {
    if (interactions.size() <= maxCapacity) {
        return;
    }
    
    // Create a vector of (index, score) pairs for scoring each memory
    std::vector<std::pair<size_t, double>> scoredMemories;
    scoredMemories.reserve(interactions.size());
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        // Calculate retention score based on importance, recency, and access patterns
        double importanceScore = interactions[i].importance;
        double recencyScore = getRecencyScore(interactions[i].timestamp);
        double accessScore = std::log1p(interactions[i].accessCount) * 0.2;
        
        // Calculate final retention score
        double retentionScore = importanceScore * 0.5 + recencyScore * 0.3 + accessScore * 0.2;
        
        scoredMemories.push_back({i, retentionScore});
    }
    
    // Sort by retention score (lowest first)
    std::sort(scoredMemories.begin(), scoredMemories.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Calculate how many memories to remove to get to maxCapacity
    size_t removeCount = interactions.size() - maxCapacity;
    
    // Collect indexes to remove (converting to set for O(1) lookup)
    std::unordered_set<size_t> indexesToRemove;
    for (size_t i = 0; i < removeCount; ++i) {
        indexesToRemove.insert(scoredMemories[i].first);
    }
    
    // Create new vector without the removed memories
    std::vector<MemoryRecord> newInteractions;
    newInteractions.reserve(maxCapacity);
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        if (indexesToRemove.find(i) == indexesToRemove.end()) {
            newInteractions.push_back(interactions[i]);
        }
    }
    
    // Replace interactions with filtered vector
    interactions = std::move(newInteractions);
    
    // Rebuild indexes to reflect changes
    rebuildIndexes();
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
            double similarity = calculateSemanticSimilarity(
                interactions[i].keywords, interactions[j].keywords);
            
            // If similarity is high, consider them for consolidation
            if (similarity > 0.7) {
                similarPairs.push_back({i, j});
            }
        }
    }
    
    // Process similar pairs (from newest to oldest)
    std::sort(similarPairs.begin(), similarPairs.end(),
              [this](const auto& a, const auto& b) {
                  return std::max(interactions[a.first].timestamp, interactions[a.second].timestamp) >
                         std::max(interactions[b.first].timestamp, interactions[b.second].timestamp);
              });
    
    // Set to track which memories have been consolidated
    std::unordered_set<size_t> consolidated;
    
    // New vector for consolidated memories
    std::vector<MemoryRecord> newInteractions;
    newInteractions.reserve(interactions.size());
    
    // Process similar pairs
    for (const auto& pair : similarPairs) {
        size_t i = pair.first;
        size_t j = pair.second;
        
        // Skip if either memory has already been consolidated
        if (consolidated.find(i) != consolidated.end() || 
            consolidated.find(j) != consolidated.end()) {
            continue;
        }
        
        // Mark both memories as consolidated
        consolidated.insert(i);
        consolidated.insert(j);
        
        // Merge the two memories
        MemoryRecord mergedRecord = mergeRecords(interactions[i], interactions[j]);
        newInteractions.push_back(mergedRecord);
    }
    
    // Add remaining memories that weren't consolidated
    for (size_t i = 0; i < interactions.size(); ++i) {
        if (consolidated.find(i) == consolidated.end()) {
            newInteractions.push_back(interactions[i]);
        }
    }
    
    // Replace with consolidated memories
    std::lock_guard<std::mutex> lock(memoryMutex);
    interactions = std::move(newInteractions);
    rebuildIndexes();
}

MemoryRecord Memory::mergeRecords(const MemoryRecord& record1, const MemoryRecord& record2) const {
    // Create a new record that merges information from both source records
    
    // Use the more recent input and response
    const MemoryRecord& primary = (record1.timestamp > record2.timestamp) ? record1 : record2;
    const MemoryRecord& secondary = (record1.timestamp > record2.timestamp) ? record2 : record1;
    
    // Combine keywords
    std::vector<std::string> mergedKeywords = record1.keywords;
    for (const auto& keyword : record2.keywords) {
        if (std::find(mergedKeywords.begin(), mergedKeywords.end(), keyword) == mergedKeywords.end()) {
            mergedKeywords.push_back(keyword);
        }
    }
    
    // Use max importance
    double mergedImportance = std::max(record1.importance, record2.importance);
    
    // Use most recent timestamp
    time_t mergedTimestamp = std::max(record1.timestamp, record2.timestamp);
    
    // Sum access counts
    int mergedAccessCount = record1.accessCount + record2.accessCount;
    
    // Use most recent access time
    time_t mergedLastAccess = std::max(record1.lastAccessTime, record2.lastAccessTime);
    
    // Create merged record
    MemoryRecord mergedRecord(
        primary.input,
        primary.response,
        primary.category,
        primary.sentiment,
        mergedKeywords,
        mergedImportance,
        mergedTimestamp
    );
    
    // Set access metadata
    mergedRecord.accessCount = mergedAccessCount;
    mergedRecord.lastAccessTime = mergedLastAccess;
    
    return mergedRecord;
}

void Memory::applyMemoryDecay() {
    // Apply memory decay based on age and access patterns
    // This simulates how human memories fade over time unless accessed frequently
    
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Get current time
    time_t currentTime = std::time(nullptr);
    
    for (auto& record : interactions) {
        // Calculate age in days
        double ageInDays = difftime(currentTime, record.timestamp) / (60 * 60 * 24);
        
        // Calculate time since last access in days
        double lastAccessInDays = difftime(currentTime, record.lastAccessTime) / (60 * 60 * 24);
        
        // Calculate decay factor based on age, last access, and access frequency
        // Memories that are accessed more frequently decay slower
        double ageFactor = 1.0 / (1.0 + 0.05 * ageInDays);
        double accessFactor = record.accessCount > 0 ? 
                             1.0 / (1.0 + 0.1 * lastAccessInDays / record.accessCount) : 
                             1.0 / (1.0 + 0.2 * lastAccessInDays);
        
        // Strong emotions and important memories decay slower
        double sentimentFactor = 1.0;
        if (record.sentiment == "very_positive" || record.sentiment == "very_negative") {
            sentimentFactor = 1.2;
        } else if (record.sentiment == "positive" || record.sentiment == "negative") {
            sentimentFactor = 1.1;
        }
        
        // Apply the decay formula
        double decayFactor = ageFactor * accessFactor * sentimentFactor;
        
        // Apply decay to importance but maintain a minimum level
        record.importance = std::max(0.1, record.importance * decayFactor);
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

void Memory::updateAccessMetadata(size_t index) const {
    // Update access count and time
    if (index < interactions.size()) {
        interactions[index].accessCount++;
        interactions[index].lastAccessTime = std::time(nullptr);
    }
}

std::string Memory::normalizeText(const std::string& text) const {
    // Convert text to lowercase
    std::string normalized = text;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    return normalized;
}

void Memory::startBackgroundWorker() {
    // Initialize background worker thread for memory management
    isRunning = true;
    backgroundThread = std::thread(&Memory::backgroundWorker, this);
}

void Memory::stopBackgroundWorker() {
    // Signal the background worker to stop and wait for it to finish
    {
        std::unique_lock<std::mutex> lock(taskMutex);
        isRunning = false;
    }
    
    // Notify worker that it should check the isRunning flag
    taskCondition.notify_one();
    
    // Wait for worker thread to finish
    if (backgroundThread.joinable()) {
        backgroundThread.join();
    }
}

void Memory::backgroundWorker() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(taskMutex);
            
            // Wait for a task or stop signal
            taskCondition.wait(lock, [this] {
                return !taskQueue.empty() || !isRunning;
            });
            
            // Check if we should exit
            if (!isRunning && taskQueue.empty()) {
                break;
            }
            
            // Get next task
            if (!taskQueue.empty()) {
                task = std::move(taskQueue.front());
                taskQueue.pop();
            }
        }
        
        // Execute the task outside the lock
        if (task) {
            try {
                task();
            } catch (const std::exception& e) {
                std::cerr << "Error in background memory task: " << e.what() << std::endl;
            }
        }
    }
}

void Memory::enqueueBackgroundTask(std::function<void()> task) {
    // Add a task to the background processing queue
    {
        std::unique_lock<std::mutex> lock(taskMutex);
        taskQueue.push(std::move(task));
    }
    
    // Notify worker that a new task is available
    taskCondition.notify_one();
}

// Advanced memory retrieval methods
std::vector<MemoryRecord> Memory::retrieveByContext(const std::string& context, size_t maxCount) const {
    // This method retrieves memories that are contextually relevant to the given input
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Extract keywords from context
    std::vector<std::string> contextKeywords = extractKeywords(context);
    
    if (contextKeywords.empty()) {
        // Fall back to recent interactions if no keywords found
        return retrieveRecent(maxCount);
    }
    
    // Score all interactions based on contextual relevance
    std::vector<std::pair<size_t, double>> scoredInteractions;
    scoredInteractions.reserve(interactions.size());
    
    for (size_t i = 0; i < interactions.size(); ++i) {
        // Calculate multiple factors for overall relevance
        double semanticSimilarity = calculateSemanticSimilarity(
            contextKeywords, interactions[i].keywords);
        
        double recencyScore = getRecencyScore(interactions[i].timestamp);
        double importanceScore = interactions[i].importance;
        
        // Combine factors with different weights
        double relevanceScore = semanticSimilarity * 0.6 + 
                              recencyScore * 0.25 + 
                              importanceScore * 0.15;
        
        scoredInteractions.push_back({i, relevanceScore});
    }
    
    // Sort by relevance score (highest first)
    std::sort(scoredInteractions.begin(), scoredInteractions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take only the top results
    size_t resultCount = std::min(maxCount, scoredInteractions.size());
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

// Memory clustering for better organization
std::unordered_map<std::string, std::vector<MemoryRecord>> Memory::clusterMemoriesByTopic() const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Use keyword frequencies to identify major topics
    std::unordered_map<std::string, int> keywordFrequency;
    for (const auto& record : interactions) {
        for (const auto& keyword : record.keywords) {
            keywordFrequency[keyword]++;
        }
    }
    
    // Sort keywords by frequency
    std::vector<std::pair<std::string, int>> sortedKeywords;
    sortedKeywords.reserve(keywordFrequency.size());
    for (const auto& pair : keywordFrequency) {
        sortedKeywords.push_back(pair);
    }
    
    std::sort(sortedKeywords.begin(), sortedKeywords.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take the top N keywords as cluster centers
    const size_t NUM_CLUSTERS = std::min(size_t(10), sortedKeywords.size());
    std::vector<std::string> clusterKeywords;
    clusterKeywords.reserve(NUM_CLUSTERS);
    
    for (size_t i = 0; i < NUM_CLUSTERS; ++i) {
        clusterKeywords.push_back(sortedKeywords[i].first);
    }
    
    // Assign each memory to the closest cluster
    std::unordered_map<std::string, std::vector<MemoryRecord>> clusters;
    
    for (const auto& record : interactions) {
        std::string bestCluster = "";
        double bestSimilarity = -1.0;
        
        for (const auto& clusterKeyword : clusterKeywords) {
            // Check if the record contains this keyword
            bool containsKeyword = std::find(record.keywords.begin(), 
                                           record.keywords.end(), 
                                           clusterKeyword) != record.keywords.end();
            
            if (containsKeyword) {
                // Calculate similarity to this cluster
                double similarity = calculateRelevanceScore(record, clusterKeyword);
                
                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    bestCluster = clusterKeyword;
                }
            }
        }
        
        // If no clear match, put in "miscellaneous" cluster
        if (bestCluster.empty()) {
            clusters["miscellaneous"].push_back(record);
        } else {
            clusters[bestCluster].push_back(record);
        }
    }
    
    return clusters;
}

// Memory analytics methods
MemoryAnalytics Memory::analyzeMemoryPatterns() const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    MemoryAnalytics analytics;
    
    // Total interaction count
    analytics.totalInteractions = interactions.size();
    
    // Calculate metrics only if we have interactions
    if (interactions.empty()) {
        return analytics;
    }
    
    // Find time range
    time_t oldestTime = interactions[0].timestamp;
    time_t newestTime = interactions[0].timestamp;
    
    for (const auto& record : interactions) {
        oldestTime = std::min(oldestTime, record.timestamp);
        newestTime = std::max(newestTime, record.timestamp);
    }
    
    // Calculate time span in days
    analytics.timeSpanDays = difftime(newestTime, oldestTime) / (60 * 60 * 24);
    
    // Calculate average interaction importance
    double totalImportance = 0.0;
    for (const auto& record : interactions) {
        totalImportance += record.importance;
    }
    analytics.averageImportance = totalImportance / interactions.size();
    
    // Get most frequent categories
    std::unordered_map<std::string, int> categoryCounts;
    for (const auto& record : interactions) {
        categoryCounts[record.category]++;
    }
    
    // Convert to vector and sort
    std::vector<std::pair<std::string, int>> sortedCategories;
    for (const auto& pair : categoryCounts) {
        sortedCategories.push_back(pair);
    }
    
    std::sort(sortedCategories.begin(), sortedCategories.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Get top categories
    size_t categoryCount = std::min(size_t(5), sortedCategories.size());
    for (size_t i = 0; i < categoryCount; ++i) {
        analytics.topCategories.push_back(sortedCategories[i]);
    }
    
    // Get sentiment distribution
    std::unordered_map<std::string, int> sentimentCounts;
    for (const auto& record : interactions) {
        sentimentCounts[record.sentiment]++;
    }
    
    // Convert to percentages
    for (const auto& pair : sentimentCounts) {
        double percentage = 100.0 * pair.second / interactions.size();
        analytics.sentimentDistribution[pair.first] = percentage;
    }
    
    // Calculate keyword diversity (unique keywords / total keywords)
    std::unordered_set<std::string> uniqueKeywords;
    size_t totalKeywords = 0;
    
    for (const auto& record : interactions) {
        totalKeywords += record.keywords.size();
        for (const auto& keyword : record.keywords) {
            uniqueKeywords.insert(keyword);
        }
    }
    
    analytics.keywordDiversity = totalKeywords > 0 ? 
        static_cast<double>(uniqueKeywords.size()) / totalKeywords : 0.0;
    
    return analytics;
}

// Memory optimization method
void Memory::optimizeMemoryStorage() {
    // This could consolidate similar memories and organize for faster retrieval
    
    // First consolidate similar memories
    consolidateMemories();
    
    // If we're over capacity, prune less important memories
    if (interactions.size() > maxCapacity) {
        pruneMemories();
    }
    
    // Rebuild indexes with optimizations
    rebuildOptimizedIndexes();
}

void Memory::rebuildOptimizedIndexes() {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    // Clear existing indexes
    keywordIndex.clear();
    categoryIndex.clear();
    sentimentIndex.clear();
    
    // Create temporary ordered list for more efficient index construction
    std::vector<std::pair<std::string, std::vector<size_t>>> tempKeywordIndex;
    std::vector<std::pair<std::string, std::vector<size_t>>> tempCategoryIndex;
    std::vector<std::pair<std::string, std::vector<size_t>>> tempSentimentIndex;
    
    // First pass - gather all data
    for (size_t i = 0; i < interactions.size(); ++i) {
        const auto& record = interactions[i];
        
        // Process keywords
        for (const auto& keyword : record.keywords) {
            keywordIndex[keyword].push_back(i);
        }
        
        // Process category
        std::string normalizedCategory = normalizeText(record.category);
        categoryIndex[normalizedCategory].push_back(i);
        
        // Process sentiment
        std::string normalizedSentiment = normalizeText(record.sentiment);
        sentimentIndex[normalizedSentiment].push_back(i);
    }
}

std::vector<MemoryRecord> Memory::findContradictions() const {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    std::vector<MemoryRecord> contradictions;
    
    // Group records by main topic
    std::unordered_map<std::string, std::vector<size_t>> topicGroups;
    
    // First, assign each record to a topic group based on its keywords
    for (size_t i = 0; i < interactions.size(); ++i) {
        for (const auto& keyword : interactions[i].keywords) {
            topicGroups[keyword].push_back(i);
        }
    }
    
    // For each topic group, look for sentiment contradictions
    for (const auto& group : topicGroups) {
        if (group.second.size() < 2) continue; // Need at least 2 for contradiction
        
        // Check for sentiment contradictions within the group
        for (size_t i = 0; i < group.second.size(); ++i) {
            for (size_t j = i + 1; j < group.second.size(); ++j) {
                const auto& record1 = interactions[group.second[i]];
                const auto& record2 = interactions[group.second[j]];
                
                // If one is positive and one is negative, potential contradiction
                bool isContradiction = 
                    (record1.sentiment.find("positive") != std::string::npos && 
                     record2.sentiment.find("negative") != std::string::npos) ||
                    (record1.sentiment.find("negative") != std::string::npos && 
                     record2.sentiment.find("positive") != std::string::npos);
                
                if (isContradiction) {
                    // Add both records if not already added
                    if (std::find_if(contradictions.begin(), contradictions.end(),
                        [&record1](const MemoryRecord& r) { 
                            return r.input == record1.input && r.response == record1.response;
                        }) == contradictions.end()) {
                        contradictions.push_back(record1);
                    }
                    
                    if (std::find_if(contradictions.begin(), contradictions.end(),
                        [&record2](const MemoryRecord& r) { 
                            return r.input == record2.input && r.response == record2.response;
                        }) == contradictions.end()) {
                        contradictions.push_back(record2);
                    }
                }
            }
        }
    }
    
    return contradictions;
}

// Set the memory capacity
void Memory::setMaxCapacity(size_t newCapacity) {
    std::lock_guard<std::mutex> lock(memoryMutex);
    
    maxCapacity = newCapacity;
    
    // If we're over the new capacity, prune memories
    if (interactions.size() > maxCapacity) {
        pruneMemories();
    }
}