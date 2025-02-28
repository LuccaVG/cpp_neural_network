#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <random>
#include <memory>
#include <functional>
#include <unordered_map>
#include <thread>
#include <future>
#include <chrono>
#include <map>

// Advanced neural network with multiple activation functions, optimization techniques,
// parallel processing, and advanced regularization methods
class NeuralNetwork {
public:
    enum class ActivationFunction {
        SIGMOID,
        RELU,
        TANH,
        LEAKY_RELU,
        ELU,          // Exponential Linear Unit
        SWISH,        // Self-gated activation function (x * sigmoid(x))
        MISH,         // Mish activation (x * tanh(softplus(x)))
        SOFTMAX
    };
    
    enum class Optimizer {
        SGD,           // Stochastic Gradient Descent
        ADAM,          // Adaptive Moment Estimation
        RMSPROP,       // Root Mean Square Propagation
        ADAGRAD,       // Adaptive Gradient Algorithm
        ADAMAX,        // AdaMax optimization
        NADAM,         // Nesterov-accelerated Adaptive Moment Estimation
        AMSGRAD        // AMSGrad variant of Adam
    };
    
    enum class InitializationMethod {
        XAVIER,        // Xavier/Glorot initialization
        HE,            // He initialization (better for ReLU)
        LECUN,         // LeCun initialization
        NORMAL,        // Normal distribution
        UNIFORM        // Uniform distribution
    };
    
    enum class RegularizationType {
        NONE,
        L1,            // Lasso Regularization
        L2,            // Ridge Regularization
        ELASTIC_NET    // Combination of L1 and L2
    };
    
    struct TrainingOptions {
        int epochs = 1000;
        double learningRate = 0.01;
        double momentum = 0.9;
        double beta1 = 0.9;       // For Adam, first moment decay rate
        double beta2 = 0.999;     // For Adam, second moment decay rate
        double epsilon = 1e-8;    // For optimizers, small constant
        
        // Learning rate schedule
        bool useLearningRateDecay = true;
        double learningRateDecayRate = 0.1;
        int learningRateDecaySteps = 100;
        
        // Regularization parameters
        RegularizationType regularizationType = RegularizationType::L2;
        double regularizationRate = 0.0001;
        double l1Ratio = 0.5;     // For elastic net, balance between L1 and L2
        
        // Normalization and dropout
        bool useBatchNormalization = false;
        bool useDropout = false;
        double dropoutRate = 0.2;
        
        // Training mode
        bool useBatchTraining = true;
        int batchSize = 32;
        
        // Optimizer
        Optimizer optimizer = Optimizer::ADAM;
        
        // Parallel processing
        bool useParallelProcessing = true;
        int numThreads = 4;
        
        // Early stopping
        bool useEarlyStopping = true;
        int earlyStoppingPatience = 10;
        double earlyStoppingMinDelta = 0.001;
    };
    
    // Constructors
    NeuralNetwork(const std::vector<int>& layerSizes,
                 ActivationFunction hiddenActivation = ActivationFunction::RELU,
                 ActivationFunction outputActivation = ActivationFunction::SIGMOID,
                 InitializationMethod initMethod = InitializationMethod::HE);
    
    // Training and prediction
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& outputs,
               const TrainingOptions& options = TrainingOptions());
    std::vector<double> predict(const std::vector<double>& input) const;
    double evaluateAccuracy(const std::vector<std::vector<double>>& inputs,
                           const std::vector<std::vector<double>>& expectedOutputs) const;
    
    double evaluateMSE(const std::vector<std::vector<double>>& inputs,
                      const std::vector<std::vector<double>>& expectedOutputs) const;
    bool saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);
    std::string getModelSummary() const;
    
    // Enhanced evaluation methods
    double evaluateClassificationAccuracy(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& expectedOutputs,
        double threshold = 0.5) const;

    std::map<std::string, double> evaluateBinaryClassificationMetrics(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& expectedOutputs,
        double threshold = 0.5) const;

    std::vector<std::vector<int>> generateConfusionMatrix(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& expectedOutputs) const;
    
private:
    // Network architecture
    std::vector<int> layerSizes;
    ActivationFunction hiddenActivation;
    ActivationFunction outputActivation;
    
    // Weights and biases
    std::vector<std::vector<std::vector<double>>> weights;  // [layer][neuron][input]
    std::vector<std::vector<double>> biases;               // [layer][neuron]
    
    // Optimizer state variables
    std::vector<std::vector<std::vector<double>>> weightVelocities;  // For momentum
    std::vector<std::vector<double>> biasVelocities;                 // For momentum
    std::vector<std::vector<std::vector<double>>> weightMoments1;    // For Adam
    std::vector<std::vector<double>> biasMoments1;                   // For Adam
    std::vector<std::vector<std::vector<double>>> weightMoments2;    // For Adam
    std::vector<std::vector<double>> biasMoments2;                   // For Adam
    
    // Additional optimizer cache variables
    std::vector<std::vector<std::vector<double>>> weightRMSCache;    // For RMSProp
    std::vector<std::vector<double>> biasRMSCache;                   // For RMSProp
    std::vector<std::vector<std::vector<double>>> weightAdagradCache; // For Adagrad
    std::vector<std::vector<double>> biasAdagradCache;                // For Adagrad
    std::vector<std::vector<std::vector<double>>> weightMaxSquared;   // For AMSGrad
    std::vector<std::vector<double>> biasMaxSquared;                  // For AMSGrad
    std::vector<std::vector<std::vector<double>>> weightInfNorm;      // For AdaMax
    std::vector<std::vector<double>> biasInfNorm;                     // For AdaMax
    
    // Batch normalization parameters
    struct BatchNormParams {
        std::vector<double> gamma;      // Scale parameter
        std::vector<double> beta;       // Shift parameter
        std::vector<double> movingMean;
        std::vector<double> movingVar;
    };
    std::vector<BatchNormParams> batchNormLayers;
    
    // Random number generator
    mutable std::mt19937 rng;
    
    // Private implementation methods
    void initializeWeights(InitializationMethod method);
    std::vector<std::vector<double>> forwardPass(const std::vector<double>& input) const;
    std::vector<std::vector<double>> forwardPassWithDropout(const std::vector<double>& input, 
                                                          double dropoutRate,
                                                          std::vector<std::vector<bool>>& dropoutMasks);
    
    std::vector<double> applyActivation(const std::vector<double>& z, ActivationFunction activation) const;
    std::vector<double> applyActivationDerivative(const std::vector<double>& z, ActivationFunction activation) const;
    void updateWeights(const std::vector<std::vector<double>>& deltas,
                      const std::vector<std::vector<double>>& activations,
                      double learningRate, double momentum,
                      double regularizationRate, Optimizer optimizer, int epoch);
    
    // Complex activation functions
    std::vector<double> softmax(const std::vector<double>& z) const;
    std::vector<double> elu(const std::vector<double>& z) const;
    std::vector<double> swish(const std::vector<double>& z) const;
    std::vector<double> mish(const std::vector<double>& z) const;
    
    // Advanced optimization algorithms
    void applyAdamOptimizer(std::vector<std::vector<std::vector<double>>>& weightGradients,
                           std::vector<std::vector<double>>& biasGradients,
                           double learningRate, double beta1, double beta2, 
                           double epsilon, int timestep);
    
    void applyRMSPropOptimizer(
        std::vector<std::vector<std::vector<double>>>& weightGradients,
        std::vector<std::vector<double>>& biasGradients,
        double learningRate, double decayRate, double epsilon);

    void applyAdagradOptimizer(
        std::vector<std::vector<std::vector<double>>>& weightGradients,
        std::vector<std::vector<double>>& biasGradients,
        double learningRate, double epsilon);

    void applyNadamOptimizer(
        std::vector<std::vector<std::vector<double>>>& weightGradients,
        std::vector<std::vector<double>>& biasGradients,
        double learningRate, double beta1, double beta2, 
        double epsilon, int timestep);

    void applyAMSGradOptimizer(
        std::vector<std::vector<std::vector<double>>>& weightGradients,
        std::vector<std::vector<double>>& biasGradients,
        double learningRate, double beta1, double beta2, 
        double epsilon, int timestep);

    void applyAdaMaxOptimizer(
        std::vector<std::vector<std::vector<double>>>& weightGradients,
        std::vector<std::vector<double>>& biasGradients,
        double learningRate, double beta1, double beta2, int timestep);
    
    // Regularization
    void applyRegularization(std::vector<std::vector<std::vector<double>>>& weightGradients,
                            double regularizationRate,
                            RegularizationType regularizationType,
                            double l1Ratio);
    
    // Batch normalization
    std::vector<double> batchNormalize(const std::vector<double>& input, 
                                     size_t layerIndex,
                                     bool isTraining);
    
    // Parallel processing utilities
    void trainInParallel(const std::vector<std::vector<double>>& inputs,
                        const std::vector<std::vector<double>>& outputs,
                        const TrainingOptions& options);
    
    // Backpropagation
    std::vector<std::vector<std::vector<double>>> backpropagate(
        const std::vector<double>& input,
        const std::vector<double>& output,
        const std::vector<std::vector<double>>& activations);
        
    // Performance-optimized batch activations
    std::vector<double> computeActivationBatch(
        const std::vector<double>& inputs, ActivationFunction activation);
        
    // Helper methods
    std::string activationToString(ActivationFunction activation) const;
};

#endif // NEURAL_NETWORK_H