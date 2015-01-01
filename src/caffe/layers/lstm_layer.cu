#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LstmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();
  Dtype* pre_gate_data = pre_gate_.mutable_gpu_data();
  Dtype* gate_data = gate_.mutable_gpu_data();
  Dtype* cell_data = cell_.mutable_gpu_data();
  Dtype* tanh_cell_data = tanh_cell_.mutable_gpu_data();

  // Initialize previous state
  caffe_copy(H_, next_cell_.gpu_data(), prev_cell_.mutable_gpu_data());
  caffe_copy(H_, next_out_.gpu_data(), prev_out_.mutable_gpu_data());

  // Compute input to hidden forward propagation
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, T_, 4*H_, I_, (Dtype)1.,
      bottom_data, weight_i, (Dtype)0., pre_gate_data);

  // Add bias 
  if (bias_term_) {
    const Dtype* bias = this->blobs_[2]->gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_, 4*H_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), bias, (Dtype)1., pre_gate_data);
  }

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    Dtype* h_t = top_data + t*H_;
    Dtype* c_t = cell_data + t*H_;
    Dtype* tanh_c_t = tanh_cell_data + t*H_;
    Dtype* i_t = gate_data + t*4*H_;
    Dtype* f_t = gate_data + t*4*H_ + 1*H_; 
    Dtype* o_t = gate_data + t*4*H_ + 2*H_; 
    Dtype* g_t = gate_data + t*4*H_ + 3*H_; 
    Dtype* pre_i_t = pre_gate_data + t*4*H_;
    Dtype* pre_g_t = pre_gate_data + t*4*H_ + 3*H_;
    Dtype* ig = ig_.mutable_gpu_data();

    // Add hidden-to-hidden propagation
    const Dtype* h_t_1 = t > 0 ? (h_t - H_) : prev_out_.gpu_data();
    caffe_gpu_gemv<Dtype>(CblasNoTrans, 4*H_, H_, (Dtype)1.,
        weight_h, h_t_1, (Dtype)1., pre_i_t);

    // Apply nonlinearity
    // Sigmoid - input/forget/output gate
    // TanH - modulation gate
    sigmoid_->Forward_gpu(3*H_, pre_i_t, i_t);
    tanh_->Forward_gpu(H_, pre_g_t, g_t);

    // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
    const Dtype* c_t_1 = t > 0 ? (c_t - H_) : prev_cell_.gpu_data();
    caffe_gpu_mul<Dtype>(H_, f_t, c_t_1, c_t);
    caffe_gpu_mul<Dtype>(H_, i_t, g_t, ig);
    caffe_gpu_add<Dtype>(H_, c_t, ig, c_t);

    // Compute output 
    tanh_->Forward_gpu(H_, c_t, tanh_c_t);
    caffe_gpu_mul<Dtype>(H_, o_t, tanh_c_t, h_t);
  }

  // Preserve cell state and output value
  caffe_copy(H_, cell_data + (T_-1)*H_, next_cell_.mutable_gpu_data());
  caffe_copy(H_, top_data + (T_-1)*H_, next_out_.mutable_gpu_data());
}

template <typename Dtype>
void LstmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();
  const Dtype* gate_data = gate_.gpu_data();
  const Dtype* cell_data = cell_.gpu_data();
  const Dtype* tanh_cell_data = tanh_cell_.gpu_data();

  Dtype* top_diff = top[0]->mutable_gpu_diff();
  Dtype* pre_gate_diff = pre_gate_.mutable_gpu_diff();
  Dtype* gate_diff = gate_.mutable_gpu_diff();
  Dtype* cell_diff = cell_.mutable_gpu_diff();

  for (int t = T_-1; t >= 0; --t) {
    Dtype* dh_t = top_diff + t*H_;
    const Dtype* c_t = cell_data + t*H_;
    Dtype* dc_t = cell_diff + t*H_;
    const Dtype* tanh_c_t = tanh_cell_data + t*H_; 
    const Dtype* i_t = gate_data + t*4*H_;
    const Dtype* f_t = gate_data + t*4*H_ + 1*H_; 
    const Dtype* o_t = gate_data + t*4*H_ + 2*H_; 
    const Dtype* g_t = gate_data + t*4*H_ + 3*H_; 
    Dtype* di_t = gate_diff + t*4*H_;
    Dtype* df_t = gate_diff + t*4*H_ + 1*H_; 
    Dtype* do_t = gate_diff + t*4*H_ + 2*H_; 
    Dtype* dg_t = gate_diff + t*4*H_ + 3*H_; 
    Dtype* pre_di_t = pre_gate_diff + t*4*H_;
    Dtype* pre_dg_t = pre_gate_diff + t*4*H_ + 3*H_;
    Dtype* fdc = fdc_.mutable_gpu_data();

    // Output gate : tanh(c(t)) * h_diff(t)
    caffe_gpu_mul<Dtype>(H_, tanh_c_t, dh_t, do_t);

    // Cell state : o(t) * tanh'(c(t)) * h_diff(t) + f(t+1) * c_diff(t+1)
    caffe_gpu_mul<Dtype>(H_, o_t, dh_t, dc_t);
    tanh_->Backward_gpu(H_, tanh_c_t, dc_t, dc_t);
    if (t < T_-1) {
      caffe_gpu_mul<Dtype>(H_, f_t + 4*H_, dc_t + H_, fdc);
      caffe_gpu_add<Dtype>(H_, fdc, dc_t, dc_t);
    }

    // Forget gate : c(t-1) * c_diff(t)
    const Dtype* c_t_1 = t > 0 ? (c_t - H_) : prev_cell_.gpu_data();
    caffe_gpu_mul<Dtype>(H_, c_t_1, dc_t, df_t);

    // Input gate : g(t) * c_diff(t)
    caffe_gpu_mul<Dtype>(H_, g_t, dc_t, di_t);
    // Input modulation gate : i(t) * c_diff(t)
    caffe_gpu_mul<Dtype>(H_, i_t, dc_t, dg_t);

    // Compute derivate before nonlinearity
    sigmoid_->Backward_gpu(3*H_, i_t, di_t, pre_di_t);
    tanh_->Backward_gpu(H_, g_t, dg_t, pre_dg_t);

    // Clip deriviates before nonlinearity
    if (clipping_threshold_ > 0.0f) {
      caffe_gpu_bound<Dtype>(4*H_, pre_di_t, -clipping_threshold_, 
          clipping_threshold_, pre_di_t);
    }
    
    if (t > 0) {
      // Backprop output errors to the previous time step
      caffe_gpu_gemv<Dtype>(CblasTrans, 4*H_, H_, (Dtype)1.,
        weight_h, pre_di_t, (Dtype)1., dh_t - H_);
    }
  }
 
  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4*H_, I_, T_, (Dtype)1.,
        pre_gate_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // Gradient w.r.t. hidden-to-hidden weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4*H_, H_, T_-1, (Dtype)1.,
        pre_gate_diff + 4*H_, top_data, (Dtype)0., this->blobs_[1]->mutable_gpu_diff());

    // Add Gradient from previous time-step
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4*H_, H_, 1, (Dtype)1.,
        pre_gate_diff, prev_out_.gpu_data(), (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[2]) { 
    // Gradient w.r.t. bias
    caffe_gpu_gemv<Dtype>(CblasTrans, T_, 4*H_, (Dtype)1., pre_gate_diff,
        bias_multiplier_.gpu_data(), (Dtype)0.,
        this->blobs_[2]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient w.r.t. bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_, I_, 4*H_, (Dtype)1.,
        pre_gate_diff, weight_i, (Dtype)0., (*bottom)[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_CLASS(LstmLayer);

}  // namespace caffe
