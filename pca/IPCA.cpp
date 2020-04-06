#include "IPCA.hpp"
#include <algorithm>
#include <vector>
#include <cassert>
#include <iostream>

using namespace Eigen;
using namespace std;

CCIPCA::CCIPCA(int dim_subspace_, int dim_data_, int l_) {
  dim_subspace = dim_subspace_;
  dim_data = dim_data_;
  num_data_points = 0;
  mean = Matrix<float, Dynamic, 1>(dim_data_);
  mean.setZero();
  eigen_vecs = Matrix<float, Dynamic, Dynamic>(dim_data_, dim_subspace_);
  l = l_;
}
CCIPCA::CCIPCA() {

}


void CCIPCA::update(Map<VectorXf>& new_face) {
  //cout << eigen_vecs << endl;

  float w1;
  float w2;

  if (num_data_points <= l) {
    w1 = ((float) (num_data_points + 1))/((float) (num_data_points + 2));
    w2 = ((float) 1) / ((float) (num_data_points+2));
  } else {
    w1 = ((float) (num_data_points + 2 - l))/((float) (num_data_points + 2));
    w2 = ((float) (1+l)) / ((float) (num_data_points+2));
  }
  mean = w1*mean + w2*new_face;
  VectorXf face = new_face - mean;
  for (int i = 0; i < dim_subspace; i++) {
    if (i > num_data_points) {
    } else if (i == num_data_points) {
      eigen_vecs.col(i) = face;
      eigen_vecs.col(i) = eigen_vecs.col(i)/eigen_vecs.col(i).norm();
    } else {
      eigen_vecs.col(i) = w1*eigen_vecs.col(i) + w2*face*face.dot(eigen_vecs.col(i))/eigen_vecs.col(i).norm();
      face = face - face.dot(eigen_vecs.col(i))*eigen_vecs.col(i)/eigen_vecs.col(i).dot(eigen_vecs.col(i));
      eigen_vecs.col(i) = eigen_vecs.col(i)/eigen_vecs.col(i).norm();
    }
  }
  num_data_points++;

  //cout << eigen_vecs << endl;
}

VectorXf CCIPCA::transform(Map<VectorXf>& new_face) {
  VectorXf transformed = new_face - mean;
  transformed = eigen_vecs.transpose()*transformed;
  return transformed;
}



float CCIPCA::distance(VectorXf& v1, VectorXf& v2) {
  VectorXf dif = v1-v2;
  return sqrt(dif.dot(dif));
}
