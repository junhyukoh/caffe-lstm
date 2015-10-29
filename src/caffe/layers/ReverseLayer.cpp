#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
void ReverseLayer<Dtype>::Reshape( const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top )
{
	top[0]->ReshapeLike( *bottom[0] );
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top )
{
	const int T_ = this->layer_param_.reshape_param().shape().dim( 0 );
	const int data_count = bottom[0]->count() / T_;
	const Dtype* bottom_data = bottom[0]->cpu_data( );
	Dtype* top_data = top[0]->mutable_cpu_data( );
	for( int t = 0; t < T_; ++t ) {
		const int bottom_offset = t * data_count;
		const int top_offset = (T_ - t -1) * data_count;
		caffe_copy<Dtype>( data_count, bottom_data + bottom_offset, top_data + top_offset );
	}
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom )
{
	if( !propagate_down[0] ) {
		return;
	}
	const int T_ = this->layer_param_.reshape_param( ).shape( ).dim( 0 );
	const int data_count = bottom[0]->count( ) / T_;
	const Dtype* top_diff = top[0]->cpu_diff( );
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff( );
	for( int t = 0; t < T_; ++t ) {
		const int bottom_offset = t * data_count;
		const int top_offset = (T_ - t - 1) * data_count;
		caffe_copy<Dtype>( data_count, top_diff + top_offset, bottom_diff + bottom_offset );
	}
}

INSTANTIATE_CLASS( ReverseLayer );
REGISTER_LAYER_CLASS( Reverse );

}  // namespace caffe
