#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe
{

template <typename Dtype>
void PermutationLayer<Dtype>::Reshape( const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top )
{
	const int T_ = bottom[0]->shape( 0 );
	const int N_ = bottom[0]->shape( 1 );
	const int C_ = bottom[0]->shape( 2 );
	vector<int> top_shape;
	top_shape.push_back( N_ );
	top_shape.push_back( C_ );
	top_shape.push_back( T_ );
	top[0]->Reshape( top_shape );
}

template <typename Dtype>
void PermutationLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top )
{
	const int T_ = bottom[0]->shape( 0 );
	const int N_ = bottom[0]->shape( 1 );
	const int C_ = bottom[0]->shape( 2 );
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	for( int t = 0; t < T_; ++t ) {
		for( int n = 0; n < N_; ++n ) {
			for( int c = 0; c < C_; c++ ) {
				top_data[top[0]->offset( n, c, t )] = *bottom_data;
				++bottom_data;
			}
		}
	}
}

template <typename Dtype>
void PermutationLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom )
{
	if( !propagate_down[0] ) {
		return;
	}
	const int T_ = bottom[0]->shape( 0 );
	const int N_ = bottom[0]->shape( 1 );
	const int C_ = bottom[0]->shape( 2 );
	const Dtype* top_diff = top[0]->cpu_diff( );
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff( );
	for( int n = 0; n < N_; ++n ) {
		for( int c = 0; c < C_; ++c ) {
			for( int t = 0; t < T_; t++ ) {
				bottom_diff[bottom[0]->offset( t, n, c )] = *top_diff;
				++top_diff;
			}
		}
	}
}

INSTANTIATE_CLASS( PermutationLayer );
REGISTER_LAYER_CLASS( Permutation );

}  // namespace caffe
