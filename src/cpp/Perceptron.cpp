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

template<typename T>
std::tuple<Eigen::Array<T, Eigen::Dynamic, 1>, double> Perceptron::init(const Eigen::MatrixXd& matrix)
{
    int numberOfRaw = matrix.rows();

    Eigen::Array<T, Eigen::Dynamic, 1> weights(numberOfRaw); // memory allocation

    for (size_t i = 0; i < numberOfRaw; i++)
    {
        weights(i) = static_cast<T> (2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0);
    }

    double b = static_cast<double>(rand()) / RAND_MAX;

    return std::make_tuple(weights, b);
}

template<typename T>
Eigen::Array<T, Eigen::Dynamic, 1> Perceptron::model(const Eigen::MatrixXd& matrix, Eigen::Array<T, Eigen::Dynamic, 1>& weights, double b)
{
    // TODO:
}