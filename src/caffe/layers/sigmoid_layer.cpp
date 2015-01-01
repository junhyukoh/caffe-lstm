#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype> 
void SigmoidLayer<Dtype>::Forward_cpu(int N, 
    const Dtype* bottom, Dtype* top) {
  for (int i = 0; i < N; ++i) {
    top[i] = sigmoid(bottom[i]);
  }
}

template <typename Dtype> 
void SigmoidLayer<Dtype>::Backward_cpu(int N, 
    const Dtype* top_data, const Dtype* top_diff, Dtype* bottom_diff) {
  for (int i = 0; i < N; ++i) {
    const Dtype sigmoid_x = top_data[i];
    bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  Forward_cpu(count, bottom_data, top_data);
  /*
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
  */
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();

    Backward_cpu(count, top_data, top_diff, bottom_diff);
    /*
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
    */
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
