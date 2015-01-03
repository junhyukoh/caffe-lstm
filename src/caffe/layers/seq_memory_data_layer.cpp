// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/data_layers.hpp"
#include <iostream>

using namespace std;
using std::string;

namespace caffe {

template <typename Dtype>
void SeqMemoryDataLayer<Dtype>::DataFetch(const Datum& datum, bool sequence_head) {
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  this->data_transformer_.Transform(0, datum, this->mean_, top_data);
  sequence_head_ = sequence_head;
}

template <typename Dtype>
void SeqMemoryDataLayer<Dtype>::DataFetch(const vector<Datum>& data, 
    const vector<vector<Dtype> >& label, bool sequence_head) {

  // LOG(ERROR) << "Datum Add ";
  const int label_channel = 
    this->layer_param_.memory_data_param().label_channels();
  const int label_width = 
    this->layer_param_.memory_data_param().label_width();
  const int label_height = 
    this->layer_param_.memory_data_param().label_height();
  const int label_size = label_channel * label_width * label_height;
  const int batch_size = this->layer_param_.memory_data_param().batch_size();
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();

  CHECK_EQ(data.size(), batch_size) << "Batch size and data size does not match";
  CHECK_EQ(label.size(), batch_size) << "Batch size and label size does not match";

  for (int i = 0; i < data.size(); ++i) {
    this->data_transformer_.Transform(i, data[i], this->mean_, top_data);

    const vector<Dtype>& l = label[i];
    CHECK_EQ(l.size(), label_size) << "Dimension of label does not match";
    for (int j = 0; j < label_size; ++j) {
      top_label[i * label_size + j] = l[j];
    }
  }

  // LOG(ERROR) << "Add Batch Sequence head" << sequence_head;
  sequence_head_ = sequence_head;
}

template <typename Dtype>
void SeqMemoryDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
}

template <typename Dtype>
void SeqMemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  BaseDataLayer<Dtype>::DataLayerSetUp(bottom, top);

  this->batch_size_ = this->layer_param_.memory_data_param().batch_size();
  this->datum_channels_ = this->layer_param_.memory_data_param().channels();
  this->datum_height_ = this->layer_param_.memory_data_param().height();
  this->datum_width_ = this->layer_param_.memory_data_param().width();
  this->datum_size_ = this->datum_channels_ * this->datum_height_ *
      this->datum_width_;
  CHECK_GT(this->batch_size_ * this->datum_size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";

  // check if we want to have mean
  if (this->transform_param_.has_mean_file()) {
    const string& mean_file = this->transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from " << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    this->data_mean_.FromProto(blob_proto);
    CHECK_EQ(this->datum_channels_, this->data_mean_.channels());
    CHECK_EQ(this->datum_height_, this->data_mean_.height());
    CHECK_EQ(this->datum_width_, this->data_mean_.width());
  }

  (*top)[0]->Reshape(this->batch_size_, this->datum_channels_,
        this->datum_height_, this->datum_width_);
  this->prefetch_data_.Reshape(this->batch_size_, this->datum_channels_, 
      this->datum_height_, this->datum_width_);
  
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // label
  if (this->output_labels_) {
    const int label_channel = 
      this->layer_param_.memory_data_param().label_channels();
    const int label_width = 
      this->layer_param_.memory_data_param().label_width();
    const int label_height = 
      this->layer_param_.memory_data_param().label_height();

    (*top)[1]->Reshape(this->batch_size_, label_channel, 
        label_height, label_width);
    this->prefetch_label_.Reshape(this->batch_size_, label_channel,
        label_height, label_width);

    LOG(INFO) << "output label size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();
  }
}

template <typename Dtype>
void SeqMemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
}

INSTANTIATE_CLASS(SeqMemoryDataLayer);

}  // namespace caffe
