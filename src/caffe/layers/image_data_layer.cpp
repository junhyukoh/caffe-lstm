#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  window_h_ = this->layer_param_.image_data_param().window_h();
  window_w_ = this->layer_param_.image_data_param().window_w();
  max_windows_h_ = this->layer_param_.image_data_param().max_windows_h();
  max_windows_w_ = this->layer_param_.image_data_param().max_windows_w();
  min_scale_ = this->layer_param_.image_data_param().min_scale();
  max_scale_ = this->layer_param_.image_data_param().max_scale();
  height_ = max_windows_h_ * window_h_;
  width_ = max_windows_w_ * window_w_;
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  LineInfo line_info;
  while( infile >> filename >> line_info.label >> line_info.ymin >> line_info.xmin >> line_info.ymax >> line_info.xmax ) {
    lines_.push_back(std::make_pair(filename, line_info));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    0, 0, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  top_shape[2] = height_;
  top_shape[3] = width_;
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  vector<int> box_shape;
  box_shape.push_back( batch_size );
  box_shape.push_back( 7 );
  top[1]->Reshape(label_shape);
  top[2]->Reshape( box_shape );
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
	this->prefetch_[i].box_.Reshape( box_shape );
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      0, 0, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  top_shape[2] = height_;
  top_shape[3] = width_;
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_box = batch->box_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
	top_shape = this->data_transformer_->InferBlobShape( cv_img );
	const int original_height = top_shape[2];
	const int original_width = top_shape[3];
	float scale;
	caffe_rng_uniform( 1, static_cast<float>(min_scale_), static_cast<float>(max_scale_), &scale );
	int new_width = int( scale * static_cast<float>(original_width) );
	int new_height = int( scale * static_cast<float>(original_height) );
	if( new_width > width_ || new_height > height_ ) {
		scale = min( static_cast<float>(width_ - 1) / static_cast<float>(original_width), static_cast<float>(height_ - 1) / static_cast<float>(original_height) );
		new_width = int( scale * static_cast<float>(original_width) );
		new_height = int( scale * static_cast<float>(original_height) );
	}
	//LOG( INFO ) << "Original height: " << original_height << " new height: " << new_height;
	//LOG( INFO ) << "Original width: " << original_width << " new width: " << new_width;
	CHECK_LE( new_height, height_ );
	CHECK_LE( new_width, width_ );
	cv::Mat new_cv_img;
	cv::resize( cv_img, new_cv_img, cv::Size( new_width, new_height ) );
    CHECK(new_cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(new_cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

	prefetch_box[7*item_id] = lines_[lines_id_].second.label;
	prefetch_box[7*item_id + 1] = lines_[lines_id_].second.ymin * scale;
	prefetch_box[7*item_id + 2] = lines_[lines_id_].second.xmin * scale;
	prefetch_box[7 * item_id + 3] = lines_[lines_id_].second.ymax * scale;
	prefetch_box[7 * item_id + 4] = lines_[lines_id_].second.xmax * scale;
	prefetch_box[7 * item_id + 5] = original_height;
	prefetch_box[7 * item_id + 6] = original_width;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
