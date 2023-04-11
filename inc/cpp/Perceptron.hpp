#pragma once

#include <Eigen/Core>

namespace ai::ml::neural
{
class Perceptron
{
public:
    Perceptron();
    ~Perceptron();

    void init();
    void model();
    void cost();

    static Eigen::MatrixXd make_blobs(int samples, int features, int centers);

    void train();
private:
    
};
    
} // namespace ai::ml::neural
