#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "neural_network.h"
#include "utils/random.h"

void loadMNISTData(const std::string& filename, std::vector<std::vector<double>>& images, std::vector<int>& labels) {
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> image(784); // 28x28 images
        int label;

        ss >> label; // First value is the label
        labels.push_back(label);

        for (int i = 0; i < 784; ++i) {
            int pixel;
            ss >> pixel;
            image[i] = pixel / 255.0; // Normalize pixel values to [0, 1]
        }
        images.push_back(image);
    }
}

int main() {
    // Load MNIST data
    std::vector<std::vector<double>> trainImages;
    std::vector<int> trainLabels;
    loadMNISTData("data/mnist_train.csv", trainImages, trainLabels);

    // Create a neural network
    NeuralNetwork nn;
    nn.addLayer(DenseLayer(784, 128, ActivationType::RELU));
    nn.addLayer(DenseLayer(128, 10, ActivationType::SOFTMAX));

    // Train the neural network
    nn.train(trainImages, trainLabels, 10, 32); // 10 epochs, batch size of 32

    // Test the neural network
    std::vector<std::vector<double>> testImages;
    std::vector<int> testLabels;
    loadMNISTData("data/mnist_test.csv", testImages, testLabels);
    double accuracy = nn.evaluate(testImages, testLabels);
    
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}