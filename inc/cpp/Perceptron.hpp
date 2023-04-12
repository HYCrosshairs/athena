#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <tuple>

namespace ai::ml::neural
{
class Perceptron
{
public:
    Perceptron();
    ~Perceptron();

    template<typename T>
    std::tuple<Eigen::Array<T, Eigen::Dynamic, 1>, double>init(const Eigen::MatrixXd& matrix);

    template<typename T>
    void model(const Eigen::MatrixXd& matrix, Eigen::Array<T, Eigen::Dynamic, 1>& weights, double b);
    
    void cost();

    static Eigen::MatrixXd make_blobs(int samples, int features, int centers);

    void train();
private:
    
};
    
} // namespace ai::ml::neural
