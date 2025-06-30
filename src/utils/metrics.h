#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <iomanip>

class Metrics {
public:
    // Classification accuracy
    static double accuracy(const std::vector<std::vector<double>>& predictions, 
                          const std::vector<std::vector<double>>& targets, 
                          double threshold = 0.5) {
        if (predictions.size() != targets.size()) return 0.0;
        
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool predClass = predictions[i][0] > threshold;
            bool trueClass = targets[i][0] > threshold;
            if (predClass == trueClass) correct++;
        }
        
        return static_cast<double>(correct) / predictions.size();
    }
    
    // Mean Squared Error
    static double meanSquaredError(const std::vector<std::vector<double>>& predictions,
                                  const std::vector<std::vector<double>>& targets) {
        if (predictions.size() != targets.size()) return 0.0;
        
        double mse = 0.0;
        int totalElements = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            for (size_t j = 0; j < predictions[i].size(); ++j) {
                double diff = predictions[i][j] - targets[i][j];
                mse += diff * diff;
                totalElements++;
            }
        }
        
        return mse / totalElements;
    }
    
    // Mean Absolute Error
    static double meanAbsoluteError(const std::vector<std::vector<double>>& predictions,
                                   const std::vector<std::vector<double>>& targets) {
        if (predictions.size() != targets.size()) return 0.0;
        
        double mae = 0.0;
        int totalElements = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            for (size_t j = 0; j < predictions[i].size(); ++j) {
                mae += std::abs(predictions[i][j] - targets[i][j]);
                totalElements++;
            }
        }
        
        return mae / totalElements;
    }
    
    // Confusion Matrix for binary classification
    struct ConfusionMatrix {
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        
        double precision() const {
            if (truePositives + falsePositives == 0) return 0.0;
            return static_cast<double>(truePositives) / (truePositives + falsePositives);
        }
        
        double recall() const {
            if (truePositives + falseNegatives == 0) return 0.0;
            return static_cast<double>(truePositives) / (truePositives + falseNegatives);
        }
        
        double f1Score() const {
            double p = precision();
            double r = recall();
            if (p + r == 0) return 0.0;
            return 2.0 * p * r / (p + r);
        }
        
        double accuracy() const {
            int total = truePositives + trueNegatives + falsePositives + falseNegatives;
            if (total == 0) return 0.0;
            return static_cast<double>(truePositives + trueNegatives) / total;
        }
        
        void print() const {
            std::cout << "Confusion Matrix:" << std::endl;
            std::cout << "              Predicted" << std::endl;
            std::cout << "           0      1" << std::endl;
            std::cout << "Actual 0 " << trueNegatives << "     " << falsePositives << std::endl;
            std::cout << "       1 " << falseNegatives << "     " << truePositives << std::endl;
            std::cout << std::endl;
            std::cout << "Precision: " << precision() << std::endl;
            std::cout << "Recall: " << recall() << std::endl;
            std::cout << "F1-Score: " << f1Score() << std::endl;
            std::cout << "Accuracy: " << accuracy() << std::endl;
        }
    };
    
    static ConfusionMatrix confusionMatrix(const std::vector<std::vector<double>>& predictions,
                                          const std::vector<std::vector<double>>& targets,
                                          double threshold = 0.5) {
        ConfusionMatrix cm;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool predClass = predictions[i][0] > threshold;
            bool trueClass = targets[i][0] > threshold;
            
            if (trueClass && predClass) cm.truePositives++;
            else if (!trueClass && !predClass) cm.trueNegatives++;
            else if (!trueClass && predClass) cm.falsePositives++;
            else if (trueClass && !predClass) cm.falseNegatives++;
        }
        
        return cm;
    }
    
    // Cross-validation
    static std::vector<double> crossValidate(class NeuralNetwork& model, 
                                           const std::vector<std::vector<double>>& features,
                                           const std::vector<std::vector<double>>& labels,
                                           int kFolds = 5) {
        std::vector<double> scores;
        size_t foldSize = features.size() / kFolds;
        
        for (int fold = 0; fold < kFolds; ++fold) {
            // Split data into train and validation
            std::vector<std::vector<double>> trainX, trainY, valX, valY;
            
            for (size_t i = 0; i < features.size(); ++i) {
                if (i >= fold * foldSize && i < (fold + 1) * foldSize) {
                    valX.push_back(features[i]);
                    valY.push_back(labels[i]);
                } else {
                    trainX.push_back(features[i]);
                    trainY.push_back(labels[i]);
                }
            }
            
            // Train model on training fold
            // Note: This would require resetting the model weights
            // model.reset(); // Would need to implement this
            // model.fit(trainX, trainY, epochs, batchSize);
            
            // Evaluate on validation fold
            std::vector<std::vector<double>> predictions;
            for (const auto& sample : valX) {
                predictions.push_back(model.predict(sample));
            }
            
            double score = accuracy(predictions, valY);
            scores.push_back(score);
        }
        
        return scores;
    }
    
    // Print comprehensive evaluation summary
    static void printEvaluationSummary(const std::vector<std::vector<double>>& predictions, 
                                      const std::vector<std::vector<double>>& targets) {
        std::cout << "\n=== Model Evaluation Summary ===" << std::endl;
        
        try {
            double acc = accuracy(predictions, targets);
            std::cout << "Accuracy: " << std::fixed << std::setprecision(4) << acc << std::endl;
        } catch (...) {
            std::cout << "Accuracy: N/A" << std::endl;
        }
        
        try {
            double mse = meanSquaredError(predictions, targets);
            std::cout << "MSE: " << std::fixed << std::setprecision(6) << mse << std::endl;
            std::cout << "RMSE: " << std::fixed << std::setprecision(6) << std::sqrt(mse) << std::endl;
        } catch (...) {
            std::cout << "MSE/RMSE: N/A" << std::endl;
        }
        
        try {
            double mae = meanAbsoluteError(predictions, targets);
            std::cout << "MAE: " << std::fixed << std::setprecision(6) << mae << std::endl;
        } catch (...) {
            std::cout << "MAE: N/A" << std::endl;
        }
        
        // Try binary classification metrics
        if (!predictions.empty() && predictions[0].size() == 1) {
            try {
                ConfusionMatrix cm = confusionMatrix(predictions, targets);
                std::cout << "\nBinary Classification Metrics:" << std::endl;
                cm.print();
            } catch (...) {
                // Not binary classification or error
            }
        }
        
        std::cout << "==============================\n" << std::endl;
    }
};

#endif // METRICS_H
