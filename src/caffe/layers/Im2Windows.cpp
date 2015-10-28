#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
void Im2WindowsLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top )
{
	window_h_ = this->layer_param_.im2windows_param( ).window_h();
	window_w_ = this->layer_param_.im2windows_param( ).window_w( );
	labels_count_ = this->layer_param_.im2windows_param().labels_count();
	reversed_ = this->layer_param_.im2windows_param().reversed();
}

template <typename Dtype>
void Im2WindowsLayer<Dtype>::Reshape( const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top )
{
	const int batch_size = bottom[0]->shape( 0 );
	const int in_channels = bottom[0]->shape( 1 );
	const int out_channels = (bottom[0]->shape( 2 ) / window_h_) * (bottom[0]->shape( 3 ) / window_w_);
	vector<int> top_shape;
	top_shape.push_back( batch_size * out_channels );
	top_shape.push_back( in_channels );
	top_shape.push_back( window_h_ );
	top_shape.push_back( window_w_ );
	top[0]->Reshape( top_shape );
	if( bottom.size() > 1 ) {
		vector<int> top_label_shape;
		top_label_shape.push_back( batch_size * out_channels );
		top_label_shape.push_back( labels_count_ );
		vector<int> top_clip_shape( 1, batch_size * out_channels );
		top[1]->Reshape( top_label_shape );
		top[2]->Reshape( top_clip_shape );
		top[3]->Reshape( top_clip_shape );
	}
}

template <typename Dtype>
Dtype GetIntersectionRatio( int ymin_window, int xmin_window, int ymax_window, int xmax_window,
	int ymin_object, int xmin_object, int ymax_object, int xmax_object )
{
	int ymin_inter = max( ymin_window, ymin_object );
	int xmin_inter = max( xmin_window, xmin_object );
	int ymax_inter = min( ymax_window, ymax_object );
	int xmax_inter = min( xmax_window, xmax_object );
	if( ymin_inter >= ymax_inter || xmin_inter >= xmax_inter ) {
		return Dtype( 0.0 );
	}
	Dtype interBoxArea = static_cast<Dtype>((ymax_inter - ymin_inter) * (xmax_inter - xmin_inter));
	Dtype windowArea = static_cast<Dtype>((ymax_window - ymin_window) * (xmax_window - xmin_window));
	return interBoxArea / windowArea;
}

template <typename Dtype>
void Im2WindowsLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top )
{
	const int batch_size = bottom[0]->shape( 0 );
	const int out_channels = (bottom[0]->shape( 2 ) / window_h_) * (bottom[0]->shape( 3 ) / window_w_);
	if( top.size() > 1 ) {
		caffe_set( top[1]->count( ), Dtype( 0.0 ), top[1]->mutable_cpu_data( ) );
	}
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* top_data_label = 0;
	Dtype* top_data_clip = 0;
	Dtype* top_data_clip_w = 0;
	const Dtype* bottom_data_label = 0;
	if( bottom.size() > 1 ) {
		top_data_label = top[1]->mutable_cpu_data();
		bottom_data_label = bottom[1]->cpu_data();
		top_data_clip = top[2]->mutable_cpu_data();
		top_data_clip_w = top[3]->mutable_cpu_data();
	}
	const Dtype* bottom_data = bottom[0]->cpu_data();
	for( int n = 0; n < bottom[0]->num(); n++ ) {
		for( int c = 0; c < bottom[0]->channels(); c++ ) {
			for( int h = 0; h < bottom[0]->height(); h++ ) {
				for( int w = 0; w < bottom[0]->width(); w++, bottom_data++ ) {
					const int top_w = w % window_w_;
					const int top_h = h % window_h_;
					int window_x = w / window_w_;
					int window_y = h / window_h_;
					if( reversed_ ) {
						window_x = (bottom[0]->shape( 3 ) / window_w_) - window_x - 1;
						window_y = (bottom[0]->shape( 2 ) / window_h_) - window_y - 1;
						CHECK_GE( window_x, 0 );
						CHECK_GE( window_y, 0 );
					}
					const int top_n = batch_size * ((bottom[0]->shape( 3 ) / window_w_) * window_y + window_x) + n;
					top_data[top[0]->offset( top_n, c, top_h, top_w )] = *bottom_data;
					if( bottom.size() > 1 ) {
						const int xmin_window = window_x * window_w_;
						const int xmax_window = xmin_window + window_w_;
						const int ymin_window = window_y * window_h_;
						const int ymax_window = ymin_window + window_h_;
						const int offset = bottom[1]->offset( n );
						const int ymin_object = int( bottom_data_label[offset + 1] );
						const int xmin_object = int( bottom_data_label[offset + 2] );
						const int ymax_object = int( bottom_data_label[offset + 3] );
						const int xmax_object = int( bottom_data_label[offset + 4] );
						const int original_height = int( bottom_data_label[offset + 5] );
						const int original_width = int( bottom_data_label[offset + 6] );
						const int label = int( bottom_data_label[offset] );
						CHECK_LT( label, labels_count_ );
						Dtype intersectionRatio = GetIntersectionRatio<Dtype>( ymin_window, xmin_window, ymax_window, xmax_window,
							ymin_object, xmin_object, ymax_object, xmax_object );
						const int label_offset = top[1]->offset( top_n, label );
						top_data_label[label_offset] = intersectionRatio;
						if( xmax_window <= original_width && ymax_window <= original_height ) {
							if( window_x == 0 ) {
								top_data_clip[top_n] = Dtype( 0.0 );
							} else {
								top_data_clip[top_n] = Dtype( 1.0 );
							}

							if( window_y == 0 ) {
								top_data_clip_w[top_n] = Dtype( 0.0 );
							} else {
								top_data_clip_w[top_n] = Dtype( 1.0 );
							}
						} else {
							top_data_clip[top_n] = Dtype( 0.0 );
							top_data_clip_w[top_n] = Dtype( 0.0 );
						}
					}
				}
			}
		}
	}
}

INSTANTIATE_CLASS( Im2WindowsLayer );
REGISTER_LAYER_CLASS( Im2Windows );

}  // namespace caffe
