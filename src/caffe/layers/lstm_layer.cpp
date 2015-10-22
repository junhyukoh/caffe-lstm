#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void LstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  clipping_threshold_ = this->layer_param_.lstm_param().clipping_threshold();
  N_ = this->layer_param_.lstm_param().batch_size(); // batch_size
  H_ = this->layer_param_.lstm_param().num_output(); // number of hidden units
  I_ = bottom[0]->count() / bottom[0]->num(); // input dimension
  W_ = this->layer_param_.lstm_param().width(); // width of 2d-sequence

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.lstm_param().weight_filler()));
 
    // input-to-hidden weights
    // Intialize the weight
    vector<int> weight_shape;
    weight_shape.push_back(5*H_);
    weight_shape.push_back(I_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[0].get());

    // hidden-to-hidden weights
    // Intialize the weight
    weight_shape.clear();
    weight_shape.push_back(5*H_);
    weight_shape.push_back(H_);
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[1].get());
	this->blobs_[3].reset( new Blob<Dtype>( weight_shape ) );
	weight_filler->Fill( this->blobs_[3].get( ) );

    // If necessary, intiialize and fill the bias term
    vector<int> bias_shape(1, 5*H_);
    this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.lstm_param().bias_filler()));
    bias_filler->Fill(this->blobs_[2].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  vector<int> cell_shape;
  cell_shape.push_back(N_);
  cell_shape.push_back(H_);
  c_0_.Reshape(cell_shape);
  h_0_.Reshape(cell_shape);
  c_T_.Reshape(cell_shape);
  h_T_.Reshape(cell_shape);
  h_to_h_.Reshape(cell_shape);

  vector<int> cell_shape_w;
  cell_shape_w.push_back( W_ );
  cell_shape_w.push_back( N_ );
  cell_shape_w.push_back( H_ );
  c_0_w_.Reshape( cell_shape_w );
  h_0_w_.Reshape( cell_shape_w );
  c_T_w_.Reshape( cell_shape_w );
  h_T_w_.Reshape( cell_shape_w );

  vector<int> gate_shape;
  gate_shape.push_back(N_);
  gate_shape.push_back(5);
  gate_shape.push_back(H_);
  h_to_gate_.Reshape(gate_shape);
  h_to_gate_w_.Reshape( gate_shape );
}

template <typename Dtype>
void LstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  T_ = bottom[0]->num() / N_; // length of sequence
  CHECK_EQ(bottom[0]->num() % N_, 0) << "Input size "
    "should be multiple of batch size";
  CHECK_EQ( T_ % W_, 0 ) << "Input size "
	  "should be multiple of width";
  CHECK_EQ(bottom[0]->count() / T_ / N_, I_) << "Input size "
    "incompatible with inner product parameters.";
  vector<int> original_top_shape;
  original_top_shape.push_back(T_*N_);
  original_top_shape.push_back(H_);
  top[0]->Reshape(original_top_shape);

  // Gate initialization
  vector<int> gate_shape;
  gate_shape.push_back(T_);
  gate_shape.push_back(N_);
  gate_shape.push_back(5);
  gate_shape.push_back(H_);
  pre_gate_.Reshape(gate_shape);
  gate_.Reshape(gate_shape);
  
  vector<int> top_shape;
  top_shape.push_back(T_);
  top_shape.push_back(N_);
  top_shape.push_back(H_);
  cell_.Reshape(top_shape);
  top_.Reshape(top_shape);
  top_.ShareData(*top[0]);
  top_.ShareDiff(*top[0]);
  
  // Set up the bias multiplier
  vector<int> multiplier_shape(1, N_*T_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1), 
    bias_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void LstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top[0]->cpu_data(), top_.cpu_data());
  Dtype* top_data = top_.mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* clip = NULL;
  const Dtype* clip_w = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->cpu_data();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  if( bottom.size( ) > 2 ) {
	clip_w = bottom[2]->cpu_data( );
	CHECK_EQ( bottom[2]->num(), bottom[2]->count() );
  }
  const Dtype* weight_i = this->blobs_[0]->cpu_data();
  const Dtype* weight_h = this->blobs_[1]->cpu_data();
  const Dtype* weight_h_w = this->blobs_[3]->cpu_data( );
  const Dtype* bias = this->blobs_[2]->cpu_data();
  Dtype* pre_gate_data = pre_gate_.mutable_cpu_data();
  Dtype* gate_data = gate_.mutable_cpu_data();
  Dtype* cell_data = cell_.mutable_cpu_data();
  Dtype* h_to_gate = h_to_gate_.mutable_cpu_data();
  Dtype* h_to_gate_w = h_to_gate_w_.mutable_cpu_data();

  // Initialize previous state
  if (!clip) {
    caffe_set(c_0_.count(), Dtype(0.), c_0_.mutable_cpu_data());
    caffe_set(h_0_.count(), Dtype(0.), h_0_.mutable_cpu_data());
  }
  if( !clip_w ) {
	caffe_set(c_0_w_.count(), Dtype(0.), c_0_w_.mutable_cpu_data());
	caffe_set(h_0_w_.count(), Dtype(0.), h_0_w_.mutable_cpu_data());
  }

  // Compute input to hidden forward propagation
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 5*H_, I_, Dtype(1.),
      bottom_data, weight_i, Dtype(0.), pre_gate_data);
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 5*H_, 1, Dtype(1.),
      bias_multiplier_.cpu_data(), bias, Dtype(1.), pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    Dtype* h_t = top_data + top_.offset(t);
    Dtype* c_t = cell_data + cell_.offset(t);
    Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
    Dtype* gate_t = gate_data + gate_.offset(t);
    Dtype* h_to_gate_t = h_to_gate;
	Dtype* h_to_gate_w_t = h_to_gate_w;
    const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
	const Dtype* clip_w_t = clip_w ? clip_w + bottom[2]->offset( t ) : NULL;
    const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.cpu_data();
    const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.cpu_data();

	const Dtype* h_t_w = t >= W_ ? (h_t - top_.offset( W_ )) : (h_0_w_.cpu_data( ) + h_0_w_.offset( t ));
	const Dtype* c_t_w = t >= W_ ? (c_t - cell_.offset( W_ )) : (c_0_w_.cpu_data( ) + c_0_w_.offset( t ));

    // Hidden-to-hidden propagation from t-1
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_, 5*H_, H_, Dtype(1.), 
        h_t_1, weight_h, Dtype(0.), h_to_gate);
	// Hidden-to-hidden propagation from t - W_
	caffe_cpu_gemm( CblasNoTrans, CblasTrans, N_, 5 * H_, H_, Dtype( 1. ),
		h_t_w, weight_h_w, Dtype( 0. ), h_to_gate_w );

    for (int n = 0; n < N_; ++n) {
		const bool cont = clip_t ? clip_t[n] : (t % W_ != 0);
		const bool cont_w = clip_w_t ? clip_w_t[n] : t >= W_;
		if( cont ) {
			caffe_add( 5 * H_, pre_gate_t, h_to_gate_t, pre_gate_t );
		}
		if( cont_w ) {
			caffe_add( 5 * H_, pre_gate_t, h_to_gate_w_t, pre_gate_t );
		}
      for (int d = 0; d < H_; ++d) {
        // Apply nonlinearity
        gate_t[d] = sigmoid(pre_gate_t[d]);
        gate_t[H_ + d] = cont ? sigmoid(pre_gate_t[H_ + d]) : Dtype(0.);
        gate_t[2*H_ + d] = sigmoid(pre_gate_t[2*H_ + d]);
        gate_t[3*H_ + d] = tanh(pre_gate_t[3*H_ + d]);
		gate_t[4 * H_ + d] = cont_w ? sigmoid( pre_gate_t[4 * H_ + d] ) : Dtype( 0. );

		// Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
		c_t[d] = gate_t[H_ + d] * c_t_1[d] + gate_t[d] * gate_t[3 * H_ + d] +
			gate_t[4 * H_ + d] * c_t_w[d];
		h_t[d] = gate_t[2 * H_ + d] * tanh( c_t[d] );
      }
      
      h_t += H_;
      c_t += H_;
      c_t_1 += H_;
	  c_t_w += H_;
      pre_gate_t += 5*H_;
      gate_t += 5*H_;
      h_to_gate_t += 5*H_;
	  h_to_gate_w_t += 5 * H_;
    }
  }
  // Preserve cell state and output value for truncated BPTT
  caffe_copy( N_*H_, cell_data + cell_.offset( T_ - 1 ), c_T_.mutable_cpu_data( ) );
  caffe_copy( N_*H_, top_data + top_.offset( T_ - 1 ), h_T_.mutable_cpu_data( ) );
  caffe_copy( N_*H_*W_, cell_data + cell_.offset( T_ - W_ ), c_T_w_.mutable_cpu_data( ) );
  caffe_copy( N_*H_*W_, top_data + top_.offset( T_ - W_ ), h_T_w_.mutable_cpu_data( ) );
}

template <typename Dtype>
void LstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top_.cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* clip = NULL;
  const Dtype* clip_w = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->cpu_data();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  if( bottom.size( ) > 2 ) {
	  clip_w = bottom[2]->cpu_data( );
	  CHECK_EQ( bottom[2]->num( ), bottom[2]->count( ) );
  }
  const Dtype* weight_i = this->blobs_[0]->cpu_data();
  const Dtype* weight_h = this->blobs_[1]->cpu_data();
  const Dtype* weight_h_w = this->blobs_[3]->cpu_data( );
  const Dtype* gate_data = gate_.cpu_data();
  const Dtype* cell_data = cell_.cpu_data();

  Dtype* top_diff = top_.mutable_cpu_diff();
  Dtype* pre_gate_diff = pre_gate_.mutable_cpu_diff();
  Dtype* gate_diff = gate_.mutable_cpu_diff();
  Dtype* cell_diff = cell_.mutable_cpu_diff();
  
  caffe_copy( N_*H_*W_, c_T_w_.cpu_diff( ), cell_diff + cell_.offset( T_ - W_ ) );

  for (int t = T_-1; t >= 0; --t) {
    Dtype* dh_t = top_diff + top_.offset(t);
    Dtype* dc_t = cell_diff + cell_.offset(t);
    Dtype* gate_diff_t = gate_diff + gate_.offset(t);
    Dtype* pre_gate_diff_t = pre_gate_diff + pre_gate_.offset(t);
    Dtype* dh_t_1 = t > 0 ? top_diff + top_.offset(t-1) : h_0_.mutable_cpu_diff();
    Dtype* dc_t_1 = t > 0 ? cell_diff + cell_.offset(t-1) : c_0_.mutable_cpu_diff();
	Dtype* dh_t_w = t >= W_ ? top_diff + top_.offset( t - W_ ) : (h_0_w_.mutable_cpu_diff( ) + h_0_w_.offset( t ));
	Dtype* dc_t_w = t >= W_ ? cell_diff + cell_.offset( t - W_ ) : (c_0_w_.mutable_cpu_diff( ) + c_0_w_.offset( t ));
	const Dtype* clip_t = clip ? clip + bottom[1]->offset( t ) : NULL;
	const Dtype* clip_w_t = clip_w ? clip_w + bottom[2]->offset( t ) : NULL;
    const Dtype* c_t = cell_data + cell_.offset(t);
	const Dtype* c_t_1 = t > 0 ? cell_data + cell_.offset( t - 1 ) : c_0_.cpu_data( );
	const Dtype* c_t_w = t >= W_ ? cell_data + cell_.offset( t - W_ ) : (c_0_w_.cpu_data( ) + c_0_w_.offset( t ));
    const Dtype* gate_t = gate_data + gate_.offset(t);

    for (int n = 0; n < N_; ++n) {
		const bool cont = clip_t ? clip_t[n] : (t % W_ != 0);
		const bool cont_w = clip_w_t ? clip_w_t[n] : t >= W_;
      for (int d = 0; d < H_; ++d) {
        const Dtype tanh_c = tanh(c_t[d]);
        gate_diff_t[2*H_ + d] = dh_t[d] * tanh_c;
        dc_t[d] += dh_t[d] * gate_t[2*H_ + d] * (Dtype(1.) - tanh_c * tanh_c);
		dc_t_w[d] = cont_w ? dc_t[d] * gate_t[4 * H_ + d] : Dtype( 0. );
		if( t > T_ - W_ ) {
			dc_t_1[d] = cont ? dc_t[d] * gate_t[H_ + d] : Dtype( 0. );
		} else {
			dc_t_1[d] += cont ? dc_t[d] * gate_t[H_ + d] : Dtype( 0. );
		}
        gate_diff_t[H_ + d] = cont ? dc_t[d] * c_t_1[d] : Dtype(0.);
        gate_diff_t[d] = dc_t[d] * gate_t[3*H_ + d];
        gate_diff_t[3*H_ +d] = dc_t[d] * gate_t[d];
		gate_diff_t[4 * H_ + d] = cont_w ? dc_t[d] * c_t_w[d] : Dtype( 0. );

        pre_gate_diff_t[d] = gate_diff_t[d] * gate_t[d] * (Dtype(1.) - gate_t[d]);
        pre_gate_diff_t[H_ + d] = gate_diff_t[H_ + d] * gate_t[H_ + d] 
            * (1 - gate_t[H_ + d]);
        pre_gate_diff_t[2*H_ + d] = gate_diff_t[2*H_ + d] * gate_t[2*H_ + d] 
            * (1 - gate_t[2*H_ + d]);
        pre_gate_diff_t[3*H_ + d] = gate_diff_t[3*H_ + d] * (Dtype(1.) - 
            gate_t[3*H_ + d] * gate_t[3*H_ + d]);
		pre_gate_diff_t[4 * H_ + d] = gate_diff_t[4 * H_ + d] * gate_t[4 * H_ + d]
			* (1 - gate_t[4 * H_ + d]);
      }

      // Clip deriviates before nonlinearity
      if (clipping_threshold_ > Dtype(0.)) {
        caffe_bound(5*H_, pre_gate_diff_t, -clipping_threshold_, 
            clipping_threshold_, pre_gate_diff_t);
      }

	  dh_t += H_;
	  c_t += H_;
	  c_t_1 += H_;
	  c_t_w += H_;
	  dc_t += H_;
	  dc_t_1 += H_;
	  dc_t_w += H_;
	  gate_t += 5 * H_;
	  gate_diff_t += 5 * H_;
	  pre_gate_diff_t += 5 * H_;
    }
    
	// Backprop output errors to the previous time step
	caffe_cpu_gemm( CblasNoTrans, CblasNoTrans, N_, H_, 5 * H_,
		Dtype( 1. ), pre_gate_diff + pre_gate_.offset( t ),
		weight_h, Dtype( 0. ), h_to_h_.mutable_cpu_data( ) );
	for( int n = 0; n < N_; ++n ) {
		const bool cont = clip_t ? clip_t[n] : (t % W_ != 0);
		const Dtype* h_to_h = h_to_h_.cpu_data( ) + h_to_h_.offset( n );
		if( cont ) {
			caffe_add( H_, dh_t_1, h_to_h, dh_t_1 );
		}
	}

	// Backprop output errors to the previous line
	caffe_cpu_gemm( CblasNoTrans, CblasNoTrans, N_, H_, 5 * H_,
		Dtype( 1. ), pre_gate_diff + pre_gate_.offset( t ),
		weight_h_w, Dtype( 0. ), h_to_h_.mutable_cpu_data( ) );
	for( int n = 0; n < N_; ++n ) {
		const bool cont_w = clip_w_t ? clip_w_t[n] : t >= W_;
		const Dtype* h_to_h = h_to_h_.cpu_data( ) + h_to_h_.offset( n );
		if( cont_w ) {
			caffe_add( H_, dh_t_w, h_to_h, dh_t_w );
		}
	}
  }
 
  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 5*H_, I_, T_*N_, Dtype(1.),
        pre_gate_diff, bottom_data, Dtype(1.), this->blobs_[0]->mutable_cpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // Gradient w.r.t. hidden-to-hidden weight
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 5*H_, H_, (T_-1)*N_, Dtype(1.),
        pre_gate_diff + pre_gate_.offset(1), top_data, 
        Dtype(1.), this->blobs_[1]->mutable_cpu_diff());

    // Add Gradient from previous time-step
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 5*H_, H_, 1, Dtype(1.),
        pre_gate_diff, h_0_.cpu_data(), 
        Dtype(1.), this->blobs_[1]->mutable_cpu_diff());
  }
  if (this->param_propagate_down_[2]) { 
    // Gradient w.r.t. bias
    caffe_cpu_gemv(CblasTrans, T_*N_, 5*H_, Dtype(1.), pre_gate_diff,
        bias_multiplier_.cpu_data(), Dtype(1.),
        this->blobs_[2]->mutable_cpu_diff());
  }
  if( this->param_propagate_down_[3] ) {
	  // Gradient w.r.t. hidden-to-hidden weight
	  caffe_cpu_gemm( CblasTrans, CblasNoTrans, 5 * H_, H_, (T_ - W_)*N_, Dtype( 1. ),
		  pre_gate_diff + pre_gate_.offset( W_ ), top_data,
		  Dtype( 1. ), this->blobs_[3]->mutable_cpu_diff( ) );

	  // Add Gradient from previous time-step
	  caffe_cpu_gemm( CblasTrans, CblasNoTrans, 5 * H_, H_, W_, Dtype( 1. ),
		  pre_gate_diff, h_0_w_.cpu_data( ),
		  Dtype( 1. ), this->blobs_[3]->mutable_cpu_diff( ) );
  }
  if (propagate_down[0]) {
    // Gradient w.r.t. bottom data
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, I_, 5*H_, Dtype(1.),
        pre_gate_diff, weight_i, Dtype(0.), bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(LstmLayer);
#endif

INSTANTIATE_CLASS(LstmLayer);
REGISTER_LAYER_CLASS(Lstm);

}  // namespace caffe
