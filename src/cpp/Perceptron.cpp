#include "Perceptron.hpp"

using namespace ai::ml::neural;

Perceptron::Perceptron(/* args */)
{
}

Perceptron::~Perceptron()
{
}

void Perceptron::train()
{

}

Eigen::MatrixXd Perceptron::make_blobs(int samples, int features, int centers)
{
    Eigen::MatrixXd X(samples, features);

    Eigen::MatrixXd ctrs = Eigen::MatrixXd::Random(centers, features);

    Eigen::VectorXi labels(samples);
    for (int i = 0; i < samples; ++i)
    {
        labels(i) = rand() % centers;
    }

    for (int i = 0; i < samples; ++i)
    {
        X.row(i) = ctrs.row(labels(i)) + Eigen::MatrixXd::Random(1, features);
    }

    return X;
}