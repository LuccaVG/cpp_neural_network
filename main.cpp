#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "neural_layer.h"

class NeuralNetwork {
public:
    // Enums for activation functions and optimizers
    enum class ActivationFunction {
        SIGMOID,
        RELU,
        LEAKY_RELU,
        TANH,
        LINEAR,
        SOFTMAX
    };
    
    enum class Optimizer {
        SGD,
        MOMENTUM,
        RMSPROP,
        ADAM
    };
    
    // Training options struct
    struct TrainingOptions {
        int epochs = 100;
        double learningRate = 0.01;
        int batchSize = 32;
        Optimizer optimizer = Optimizer::SGD;
        double momentum = 0.9;           // For MOMENTUM and ADAM
        double beta1 = 0.9;              // For ADAM (momentum)
        double beta2 = 0.999;            // For ADAM (RMSProp)
        double epsilon = 1e-8;           // For ADAM and RMSProp
        bool useL1Regularization = false;
        bool useL2Regularization = false;
        double l1Lambda = 0.0;
        double l2Lambda = 0.0;
        bool useDropout = false;
        double dropoutRate = 0.0;
        bool useBatchNormalization = false;
        bool useEarlyStopping = false;
        int patienceEpochs = 10;
        double earlyStoppingDelta = 0.001;
    };
    
private:
    std::vector<NeuralLayer> layers;
    std::vector<int> layerSizes;
    ActivationFunction hiddenActivation;
    ActivationFunction outputActivation;
    
    // Optimizer state variables
    std::vector<std::vector<std::vector<double>>> velocities;    // For momentum
    std::vector<std::vector<std::vector<double>>> cacheWeights;  // For RMSProp/ADAM
    std::vector<std::vector<double>> velocitiesBias;
    std::vector<std::vector<double>> cacheBias;
    
    // Internal helper methods
    void initializeOptimizer(const TrainingOptions& options);
    void updateWeightsWithOptimizer(int layerIndex, int neuronIndex, int weightIndex, 
                                   double gradient, const TrainingOptions& options, int t);
    void updateBiasWithOptimizer(int layerIndex, int neuronIndex, 
                                double gradient, const TrainingOptions& options, int t);
    std::function<double(double)> getActivationFunction(ActivationFunction func);
    std::function<double(double)> getActivationDerivative(ActivationFunction func);
    std::vector<double> applySoftmax(const std::vector<double>& inputs);
    double calculateLoss(const std::vector<double>& outputs, const std::vector<double>& targets);
    
public:
    // Constructor with layer configuration
    NeuralNetwork(const std::vector<int>& layerSizes, 
                 ActivationFunction hiddenActivation = ActivationFunction::SIGMOID,
                 ActivationFunction outputActivation = ActivationFunction::SIGMOID);
    
    // Forward pass - predict output
    std::vector<double> predict(const std::vector<double>& inputs);
    
    // Training methods
    void train(const std::vector<std::vector<double>>& inputs, 
              const std::vector<std::vector<double>>& targets, 
              const TrainingOptions& options);
              
    double trainOnBatch(const std::vector<std::vector<double>>& batchInputs,
                       const std::vector<std::vector<double>>& batchTargets,
                       const TrainingOptions& options);
                       
    // Evaluation methods
    double evaluateAccuracy(const std::vector<std::vector<double>>& inputs, 
                           const std::vector<std::vector<double>>& targets);
                           
    double evaluateLoss(const std::vector<std::vector<double>>& inputs, 
                       const std::vector<std::vector<double>>& targets);
                       
    std::map<std::string, double> evaluateBinaryClassificationMetrics(
        const std::vector<std::vector<double>>& inputs, 
        const std::vector<std::vector<double>>& targets);
    
    // Model serialization
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
    
    // Utility methods
    std::string getModelSummary() const;
};