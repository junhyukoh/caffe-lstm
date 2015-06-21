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
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top[0]->gpu_data(), top_.gpu_data());
  Dtype* top_data = top_.mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* clip = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->gpu_data();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();
  const Dtype* bias = this->blobs_[2]->gpu_data();
  Dtype* pre_gate_data = pre_gate_.mutable_gpu_data();
  Dtype* gate_data = gate_.mutable_gpu_data();
  Dtype* cell_data = cell_.mutable_gpu_data();
  Dtype* tanh_cell_data = tanh_cell_.mutable_gpu_data();
  Dtype* ig = ig_.mutable_gpu_data();
  Dtype* mask = clip_mask_.mutable_gpu_data();

  // Initialize previous state
  if (clip) {
    // compute clip mask
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, H_, 1, (Dtype)1.,
        clip, clip_multiplier_.gpu_data(), (Dtype)0., mask);
    caffe_gpu_mul(c_0_.count(), c_T_.gpu_data(), mask, c_0_.mutable_gpu_data());
    caffe_gpu_mul(h_0_.count(), h_T_.gpu_data(), mask, h_0_.mutable_gpu_data());
  }
  else {
    caffe_gpu_set(c_0_.count(), (Dtype)0., c_0_.mutable_gpu_data());
    caffe_gpu_set(h_0_.count(), (Dtype)0., h_0_.mutable_gpu_data());
  }

  // Compute input to hidden forward propagation
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 4*H_, I_, (Dtype)1.,
      bottom_data, weight_i, (Dtype)0., pre_gate_data);

  // Add bias 
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 4*H_, 1, (Dtype)1.,
      bias_multiplier_.gpu_data(), bias, (Dtype)1., pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    Dtype* h_t = top_data + top_.offset(t);
    Dtype* c_t = cell_data + cell_.offset(t);
    Dtype* tanh_c_t = tanh_cell_data + tanh_cell_.offset(t);
    Dtype* i_t = gate_data + gate_.offset(t);
    Dtype* f_t = gate_data + gate_.offset(t, 0, 1); 
    Dtype* o_t = gate_data + gate_.offset(t, 0, 2); 
    Dtype* g_t = gate_data + gate_.offset(t, 0, 3); 
    Dtype* pre_i_t = pre_gate_data + pre_gate_.offset(t);
    Dtype* pre_g_t = pre_gate_data + pre_gate_.offset(t, 0, 3);
    Dtype* mask_t = mask + clip_mask_.offset(t);
    const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.gpu_data();
    const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.gpu_data();

    // Hidden-to-hidden propagation
    if (clip) {
      caffe_gpu_mul(N_*H_, h_t_1, mask_t, clipped_.mutable_gpu_data());
      h_t_1 = clipped_.gpu_data();
    }
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, N_, 4*H_, H_, (Dtype)1.,
        h_t_1, weight_h, (Dtype)1., pre_gate_data + pre_gate_.offset(t));

    for (int n = 0; n < N_; ++n) {
      // Apply nonlinearity
      caffe_gpu_sigmoid(3*H_, pre_i_t, i_t);
      caffe_gpu_tanh(H_, pre_g_t, g_t);
      if (clip) {
        caffe_gpu_mul(H_, f_t, mask_t, f_t);
      }

      // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
      caffe_gpu_mul(H_, f_t, c_t_1, c_t);
      caffe_gpu_mul(H_, i_t, g_t, ig);
      caffe_gpu_add(H_, c_t, ig, c_t);

      // Compute output 
      caffe_gpu_tanh(H_, c_t, tanh_c_t);
      caffe_gpu_mul(H_, o_t, tanh_c_t, h_t);
      
      h_t += H_;
      c_t += H_;
      tanh_c_t += H_;
      c_t_1 += H_;
      i_t += 4*H_;
      f_t += 4*H_;
      o_t += 4*H_;
      g_t += 4*H_;
      pre_i_t += 4*H_;
      pre_g_t += 4*H_;
      mask_t += H_;
    }
  }
  // Preserve cell state and output value for truncated BPTT
  caffe_copy(N_*H_, cell_data + cell_.offset(T_-1), c_T_.mutable_gpu_data());
  caffe_copy(N_*H_, top_data + top_.offset(T_-1), h_T_.mutable_gpu_data());
}

template <typename Dtype>
void LstmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* clip = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->gpu_data();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();
  const Dtype* gate_data = gate_.gpu_data();
  const Dtype* cell_data = cell_.gpu_data();
  const Dtype* tanh_cell_data = tanh_cell_.gpu_data();
  const Dtype* mask = clip_mask_.gpu_data();

  Dtype* top_diff = top_.mutable_gpu_diff();
  Dtype* pre_gate_diff = pre_gate_.mutable_gpu_diff();
  Dtype* gate_diff = gate_.mutable_gpu_diff();
  Dtype* cell_diff = cell_.mutable_gpu_diff();
  
  for (int t = T_-1; t >= 0; --t) {
    Dtype* dh_t = top_diff + top_.offset(t);
    const Dtype* c_t = cell_data + cell_.offset(t);
    Dtype* dc_t = cell_diff + cell_.offset(t);
    const Dtype* tanh_c_t = tanh_cell_data + tanh_cell_.offset(t); 
    const Dtype* i_t = gate_data + gate_.offset(t);
    const Dtype* f_t = gate_data + gate_.offset(t, 0, 1); 
    const Dtype* o_t = gate_data + gate_.offset(t, 0, 2); 
    const Dtype* g_t = gate_data + gate_.offset(t, 0, 3); 
    Dtype* di_t = gate_diff + gate_.offset(t);
    Dtype* df_t = gate_diff + gate_.offset(t, 0, 1); 
    Dtype* do_t = gate_diff + gate_.offset(t, 0, 2); 
    Dtype* dg_t = gate_diff + gate_.offset(t, 0, 3); 
    Dtype* pre_di_t = pre_gate_diff + pre_gate_.offset(t);
    Dtype* pre_dg_t = pre_gate_diff + pre_gate_.offset(t, 0, 3);
    Dtype* fdc = fdc_.mutable_gpu_data();
    const Dtype* mask_t = mask + clip_mask_.offset(t);

    for (int n = 0; n < N_; ++n) {
      // Output gate : tanh(c(t)) * h_diff(t)
      caffe_gpu_mul(H_, tanh_c_t, dh_t, do_t);

      // Cell state : o(t) * tanh'(c(t)) * h_diff(t) + f(t+1) * c_diff(t+1)
      caffe_gpu_mul(H_, o_t, dh_t, dc_t);
      caffe_gpu_tanh_diff(H_, tanh_c_t, dc_t, dc_t);
      if (t < T_-1) {
        caffe_gpu_mul(H_, f_t + gate_.offset(1), dc_t + cell_.offset(1), fdc);
        caffe_gpu_add(H_, fdc, dc_t, dc_t);
      }

      // Forget gate : c(t-1) * c_diff(t)
      const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.gpu_data();
      if (clip) {
        caffe_gpu_mul(H_, c_t_1, mask_t, clipped_.mutable_gpu_data());
        c_t_1 = clipped_.gpu_data();
      }
      caffe_gpu_mul(H_, c_t_1, dc_t, df_t);

      // Input gate : g(t) * c_diff(t)
      caffe_gpu_mul(H_, g_t, dc_t, di_t);
      // Input modulation gate : i(t) * c_diff(t)
      caffe_gpu_mul(H_, i_t, dc_t, dg_t);

      // Compute derivate before nonlinearity
      caffe_gpu_sigmoid_diff(3*H_, i_t, di_t, pre_di_t);
      caffe_gpu_tanh_diff(H_, g_t, dg_t, pre_dg_t);
      
      // Clip deriviates before nonlinearity
      if (clipping_threshold_ > 0.0f) {
        caffe_gpu_bound(4*H_, pre_di_t, -clipping_threshold_, 
            clipping_threshold_, pre_di_t);
      }

      dh_t += H_;
      c_t += H_;
      dc_t += H_;
      tanh_c_t += H_;
      mask_t += H_;
      di_t += 4*H_;
      df_t += 4*H_;
      do_t += 4*H_;
      dg_t += 4*H_;
      i_t += 4*H_;
      f_t += 4*H_;
      o_t += 4*H_;
      g_t += 4*H_;
      pre_di_t += 4*H_;
      pre_dg_t += 4*H_;
    }
    
    if (t > 0) {
      Dtype* dh_t_1 = top_diff + top_.offset(t-1);
      // Backprop output errors to the previous time step
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, N_, H_, 4*H_,
          (Dtype)1., pre_gate_diff + pre_gate_.offset(t), 
          weight_h, (Dtype)0., clipped_.mutable_gpu_data());
      if (clip) {
        caffe_gpu_mul(N_*H_, clipped_.gpu_data(),
          mask + clip_mask_.offset(t), 
          clipped_.mutable_gpu_data());
      }
      caffe_gpu_add(N_*H_, dh_t_1, clipped_.gpu_data(), dh_t_1);
    }
  }
 
  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, 4*H_, I_, T_*N_, (Dtype)1.,
        pre_gate_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // Gradient w.r.t. hidden-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, (T_-1)*N_, (Dtype)1.,
        pre_gate_diff + pre_gate_.offset(1), top_data, 
        (Dtype)0., this->blobs_[1]->mutable_gpu_diff());

    // Add Gradient from previous time-step
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, 1, (Dtype)1.,
        pre_gate_diff, h_0_.gpu_data(), 
        (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }
  if (this->param_propagate_down_[2]) { 
    // Gradient w.r.t. bias
    caffe_gpu_gemv(CblasTrans, T_*N_, 4*H_, (Dtype)1., pre_gate_diff,
        bias_multiplier_.gpu_data(), (Dtype)0.,
        this->blobs_[2]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient w.r.t. bottom data
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, I_, 4*H_, (Dtype)1.,
        pre_gate_diff, weight_i, (Dtype)0., bottom[0]->mutable_gpu_diff());
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(LstmLayer);

}  // namespace caffe
