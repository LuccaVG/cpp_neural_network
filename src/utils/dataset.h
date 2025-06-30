#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iostream>

struct DataPoint {
    std::vector<double> features;
    std::vector<double> labels;
};

class Dataset {
private:
    std::vector<DataPoint> data;
    std::mt19937 rng;
    
public:
    Dataset() : rng(std::random_device{}()) {}
    
    // Add a sample to the dataset
    void addSample(const std::vector<double>& features, const std::vector<double>& labels) {
        DataPoint point;
        point.features = features;
        point.labels = labels;
        data.push_back(point);
    }
    
    // Compatibility methods
    void normalize() { normalizeFeatures(); }
    std::vector<std::vector<double>> getFeatures() const {
        std::vector<std::vector<double>> features;
        for (const auto& point : data) {
            features.push_back(point.features);
        }
        return features;
    }
    
    std::vector<std::vector<double>> getLabels() const {
        std::vector<std::vector<double>> labels;
        for (const auto& point : data) {
            labels.push_back(point.labels);
        }
        return labels;
    }
    
    // Load CSV data
    bool loadCSV(const std::string& filename, int labelColumn = -1, bool hasHeader = true) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return false;
        }
        
        std::string line;
        bool isFirstLine = true;
        
        while (std::getline(file, line)) {
            if (isFirstLine && hasHeader) {
                isFirstLine = false;
                continue;
            }
            
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stod(cell));
                } catch (const std::exception&) {
                    std::cerr << "Warning: Could not parse value: " << cell << std::endl;
                    row.push_back(0.0);
                }
            }
            
            if (!row.empty()) {
                DataPoint point;
                
                if (labelColumn == -1) {
                    // Last column is label
                    point.labels.push_back(row.back());
                    row.pop_back();
                    point.features = row;
                } else {
                    // Specified column is label
                    point.labels.push_back(row[labelColumn]);
                    row.erase(row.begin() + labelColumn);
                    point.features = row;
                }
                
                data.push_back(point);
            }
        }
        
        std::cout << "Loaded " << data.size() << " samples with " 
                  << (data.empty() ? 0 : data[0].features.size()) << " features" << std::endl;
        return true;
    }
    
    // Normalize features to [0, 1]
    void normalizeFeatures() {
        if (data.empty()) return;
        
        size_t numFeatures = data[0].features.size();
        std::vector<double> minVals(numFeatures, std::numeric_limits<double>::max());
        std::vector<double> maxVals(numFeatures, std::numeric_limits<double>::lowest());
        
        // Find min and max values
        for (const auto& point : data) {
            for (size_t i = 0; i < numFeatures; ++i) {
                minVals[i] = std::min(minVals[i], point.features[i]);
                maxVals[i] = std::max(maxVals[i], point.features[i]);
            }
        }
        
        // Normalize
        for (auto& point : data) {
            for (size_t i = 0; i < numFeatures; ++i) {
                if (maxVals[i] != minVals[i]) {
                    point.features[i] = (point.features[i] - minVals[i]) / (maxVals[i] - minVals[i]);
                }
            }
        }
    }
    
    // Standardize features (mean=0, std=1)
    void standardizeFeatures() {
        if (data.empty()) return;
        
        size_t numFeatures = data[0].features.size();
        std::vector<double> means(numFeatures, 0.0);
        std::vector<double> stds(numFeatures, 0.0);
        
        // Calculate means
        for (const auto& point : data) {
            for (size_t i = 0; i < numFeatures; ++i) {
                means[i] += point.features[i];
            }
        }
        for (auto& mean : means) {
            mean /= data.size();
        }
        
        // Calculate standard deviations
        for (const auto& point : data) {
            for (size_t i = 0; i < numFeatures; ++i) {
                double diff = point.features[i] - means[i];
                stds[i] += diff * diff;
            }
        }
        for (auto& std : stds) {
            std = std::sqrt(std / data.size());
        }
        
        // Standardize
        for (auto& point : data) {
            for (size_t i = 0; i < numFeatures; ++i) {
                if (stds[i] != 0.0) {
                    point.features[i] = (point.features[i] - means[i]) / stds[i];
                }
            }
        }
    }
    
    // Split dataset into train/test
    std::pair<Dataset, Dataset> trainTestSplit(double testRatio = 0.2) {
        Dataset trainSet, testSet;
        
        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        size_t testSize = static_cast<size_t>(data.size() * testRatio);
        
        for (size_t i = 0; i < testSize; ++i) {
            testSet.data.push_back(data[indices[i]]);
        }
        
        for (size_t i = testSize; i < data.size(); ++i) {
            trainSet.data.push_back(data[indices[i]]);
        }
        
        return {trainSet, testSet};
    }
    
    // Shuffle the dataset
    void shuffle() {
        std::shuffle(data.begin(), data.end(), rng);
    }
    
    // Get batch of data
    std::vector<DataPoint> getBatch(size_t startIdx, size_t batchSize) const {
        std::vector<DataPoint> batch;
        for (size_t i = startIdx; i < std::min(startIdx + batchSize, data.size()); ++i) {
            batch.push_back(data[i]);
        }
        return batch;
    }
    
    // Getters
    size_t size() const { return data.size(); }
    const std::vector<DataPoint>& getData() const { return data; }
    
    // Convert to format expected by neural network
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> getTrainingData() const {
        std::vector<std::vector<double>> features, labels;
        for (const auto& point : data) {
            features.push_back(point.features);
            labels.push_back(point.labels);
        }
        return {features, labels};
    }
};

#endif // DATASET_H
