#ifndef MEMORY_H
#define MEMORY_H

#include <vector>
#include <string>
#include <unordered_map>
#include <ctime>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

// Memory Record struct
struct MemoryRecord {
    std::string input;
    std::string response;
    std::string category;
    std::string sentiment;
    std::vector<std::string> keywords;
    double importance;
    time_t timestamp;
    mutable int accessCount;
    mutable time_t lastAccessTime;
    
    MemoryRecord(
        const std::string& input,
        const std::string& response,
        const std::string& category,
        const std::string& sentiment,
        const std::vector<std::string>& keywords,
        double importance,
        time_t timestamp
    );
};

// Memory Analytics struct
struct MemoryAnalytics {
    size_t totalInteractions = 0;
    double timeSpanDays = 0.0;
    double averageImportance = 0.0;
    std::vector<std::pair<std::string, int>> topCategories;
    std::unordered_map<std::string, double> sentimentDistribution;
    double keywordDiversity = 0.0;
};

// Main Memory class
class Memory {
public:
    Memory(size_t capacity = 1000);
    ~Memory();
    
    // Core memory operations
    void storeInteraction(const std::string& input, const std::string& response, 
                         const std::string& category = "general",
                         const std::string& sentiment = "neutral");
    
    // Enhanced retrieval options
    std::vector<MemoryRecord> retrieveInteractions(size_t count = 10) const;
    std::vector<MemoryRecord> retrieveByKeyword(const std::string& keyword, size_t maxCount = 10) const;
    std::vector<MemoryRecord> retrieveByCategory(const std::string& category, size_t maxCount = 10) const;
    std::vector<MemoryRecord> retrieveBySentiment(const std::string& sentiment, size_t maxCount = 10) const;
    std::vector<MemoryRecord> retrieveRecent(size_t count = 10) const;
    std::vector<MemoryRecord> retrieveByImportance(size_t count = 10) const;
    std::vector<MemoryRecord> searchBySemanticSimilarity(const std::string& query, size_t count = 10) const;
    std::vector<MemoryRecord> retrieveByContext(const std::string& context, size_t maxCount = 10) const;
    std::vector<MemoryRecord> findContradictions() const;
    
    // Memory management
    void clearMemory();
    void pruneOldestRecords(size_t keepCount = 0);
    void setMaxCapacity(size_t newCapacity);
    void optimizeMemoryStorage();
    size_t size() const;
    bool isEmpty() const;
    
    // Persistence
    bool saveToFile(const std::string& filename) const;
    bool loadFromFile(const std::string& filename);
    
    // Memory analysis
    std::vector<std::pair<std::string, int>> getTopKeywords(size_t count = 10) const;
    std::vector<std::pair<std::string, int>> getCategoryCounts() const;
    std::vector<std::pair<std::string, int>> getSentimentDistribution() const;
    std::unordered_map<std::string, std::vector<MemoryRecord>> clusterMemoriesByTopic() const;
    MemoryAnalytics analyzeMemoryPatterns() const;
    
private:
    std::vector<MemoryRecord> interactions;
    size_t maxCapacity;
    
    // Thread management for background tasks
    std::thread backgroundThread;
    std::queue<std::function<void()>> taskQueue;
    std::mutex taskMutex;
    std::condition_variable taskCondition;
    bool isRunning;
    
    // Cache for improved performance
    mutable std::mutex memoryMutex;
    mutable std::unordered_map<std::string, std::vector<size_t>> keywordIndex;
    mutable std::unordered_map<std::string, std::vector<size_t>> categoryIndex;
    mutable std::unordered_map<std::string, std::vector<size_t>> sentimentIndex;
    
    // Background processing
    void startBackgroundWorker();
    void stopBackgroundWorker();
    void backgroundWorker();
    void enqueueBackgroundTask(std::function<void()> task);
    
    // Memory optimization
    void updateIndexes(const MemoryRecord& record, size_t position);
    void consolidateMemories();
    void applyMemoryDecay();
    void pruneMemories();
    void rebuildOptimizedIndexes();
    
    // Helper methods
    std::vector<std::string> extractKeywords(const std::string& text) const;
    void rebuildIndexes();
    double calculateRelevanceScore(const MemoryRecord& record, const std::string& keyword) const;
    MemoryRecord mergeRecords(const MemoryRecord& record1, const MemoryRecord& record2) const;
    
    // Added missing helper methods
    std::string normalizeText(const std::string& text) const;
    void updateAccessMetadata(size_t index) const;
    double getRecencyScore(time_t timestamp) const;
    double calculateSemanticSimilarity(
        const std::vector<std::string>& keywords1,
        const std::vector<std::string>& keywords2) const;
    double calculateImportance(const std::string& input, 
                              const std::string& category, 
                              const std::string& sentiment) const;
};

#endif // MEMORY_H