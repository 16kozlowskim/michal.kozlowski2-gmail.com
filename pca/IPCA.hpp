
#include </Users/michal/Downloads/eigen-3.3.7/Eigen/Eigen>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;

class CCIPCA {
public:


    CCIPCA(int dim_subspace_, int dim_data_, int l_);
    CCIPCA();

    void update(Eigen::Map<Eigen::VectorXf>& new_face);
    Eigen::VectorXf transform(Eigen::Map<Eigen::VectorXf>& new_face);
    Eigen::VectorXf transform(Eigen::VectorXf& new_face);
    float distance(Eigen::VectorXf& v1, Eigen::VectorXf& v2);

private:
    int num_data_points;
    int dim_data;
    int dim_subspace;
    int l;
    Eigen::Matrix<float, Eigen::Dynamic, 1> mean;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_vecs;
};
