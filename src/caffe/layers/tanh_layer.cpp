// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype> 
void TanHLayer<Dtype>::Forward_cpu(int N, 
    const Dtype* bottom, Dtype* top) {
  Dtype exp2x;
  for (int i = 0; i < N; ++i) {
    exp2x = exp(2 * bottom[i]);
    top[i] = (exp2x - Dtype(1)) / (exp2x + Dtype(1));
  }
}

template <typename Dtype> 
void TanHLayer<Dtype>::Backward_cpu(int N, 
    const Dtype* top_data, const Dtype* top_diff, Dtype* bottom_diff) {
  Dtype tanhx;
  for (int i = 0; i < N; ++i) {
    tanhx = top_data[i];
    bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
  }
}

template <typename Dtype>
void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  //Dtype exp2x;
  const int count = bottom[0]->count();
  Forward_cpu(count, bottom_data, top_data);
  /*
  for (int i = 0; i < count; ++i) {
    exp2x = exp(2 * bottom_data[i]);
    top_data[i] = (exp2x - Dtype(1)) / (exp2x + Dtype(1));
  }
  */
}

template <typename Dtype>
void TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    //Dtype tanhx;

    Backward_cpu(count, top_data, top_diff, bottom_diff);
    /*
    for (int i = 0; i < count; ++i) {
      tanhx = top_data[i];
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
    }
    */
  }
}

#ifdef CPU_ONLY
STUB_GPU(TanHLayer);
#endif

INSTANTIATE_CLASS(TanHLayer);

}  // namespace caffe
