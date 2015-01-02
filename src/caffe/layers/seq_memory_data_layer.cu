#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void SeqMemoryDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
 // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      (*top)[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        (*top)[1]->mutable_gpu_data());
  }
}

INSTANTIATE_CLASS(SeqMemoryDataLayer);

}  // namespace caffe
