#include <cuda_runtime.h>
#include <ctime>
#include "caffe/caffe.hpp"
#include <math.h>
#include <iostream>

using namespace caffe;
using namespace std;


double f_x(double t) {
  return 0.5*sin(2*t) - 0.05*cos(17*t + 0.8) 
      + 0.05*sin(25*t+10) - 0.02*cos(45*t + 0.3);
}

int main(int argc, char** argv)
{
  if (argc < 4) {  
    LOG(ERROR) << "lstm_sequence solver_proto(1) result_path(2) sequence_length(3)";
    return 0;
  }

  // Converting input parameters
  const char* net_solver(argv[1]);
  const char* result_path(argv[2]);
  const int TotalLength = atoi(argv[3]);

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(net_solver, &solver_param);

  shared_ptr<Solver<double> > solver;
  solver.reset(GetSolver<double>(solver_param));
  shared_ptr<Net<double> > train_net(solver->net());
  shared_ptr<Net<double> > test_net(new Net<double> (solver_param.net(), TEST));
  CHECK(train_net->has_layer("data"));
  CHECK(train_net->has_layer("clip"));
  CHECK(test_net->has_layer("data"));
  CHECK(test_net->has_layer("clip"));
  MemoryDataLayer<double>* train_data_layer = 
    static_cast<MemoryDataLayer<double>*>(train_net->layer_by_name("data").get());
  MemoryDataLayer<double>* train_clip_layer = 
    static_cast<MemoryDataLayer<double>*>(train_net->layer_by_name("clip").get()); 
  MemoryDataLayer<double>* test_data_layer = 
    static_cast<MemoryDataLayer<double>*>(test_net->layer_by_name("data").get());
  MemoryDataLayer<double>* test_clip_layer = 
    static_cast<MemoryDataLayer<double>*>(test_net->layer_by_name("clip").get());
  const LayerParameter& layer_param = train_data_layer->layer_param();
  int seq_length = layer_param.memory_data_param().batch_size();
  CHECK_EQ(TotalLength % seq_length, 0);

  // Initialize bias for the forget gate to 5 as described in the clockwork RNN paper
  const vector<shared_ptr<Layer<double> > >& layers = train_net->layers();
  for (int i = 0; i < layers.size(); ++i) {
    if (strcmp(layers[i]->type(), "Lstm") != 0) {
      continue;
    }
    const int h = layers[i]->layer_param().lstm_param().num_output();
    shared_ptr<Blob<double> > bias = layers[i]->blobs()[2];
    caffe_set(h, 5.0, bias->mutable_cpu_data() + h);
  }

  // Set device id and mode
  if (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    LOG(INFO) << "Use GPU with device ID " << solver_param.device_id();
    Caffe::SetDevice(solver_param.device_id());
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  vector<int> sequence_shape(1, TotalLength);
  vector<int> data_shape(1, seq_length);
  Blob<double> sequence(sequence_shape);
  Blob<double> data(data_shape);
  Blob<double> clip(data_shape);
  caffe_set(seq_length, 1.0, clip.mutable_cpu_data());

  // Construct data 
  double mean = 0;
  double max_abs = 0;
  for (int i = 0; i < TotalLength; ++i) {
    double val = f_x(i * 0.01);
    max_abs = max(max_abs, abs(val));
  }
  for (int i = 0; i < TotalLength; ++i) {
    mean += f_x(i * 0.01) / max_abs;
  }
  mean /= TotalLength;
  for (int i = 0; i < TotalLength; ++i) {
    sequence.mutable_cpu_data()[i] = f_x(i*0.01) / max_abs - mean;
  }

  // Training
  int iter = 0;
  double dummy;
  while(iter < solver_param.max_iter()) {
    int seq_idx = iter % (TotalLength / seq_length);
    clip.mutable_cpu_data()[0] = seq_idx == 0 ? 0.0 : 1.0;
    train_data_layer->Reset(data.mutable_cpu_data(), 
      sequence.mutable_cpu_data() + sequence.offset(seq_idx * seq_length), 
      seq_length);
    train_clip_layer->Reset(clip.mutable_cpu_data(), &dummy, seq_length);
    solver->Step(1);
    iter++;
  }

  // Output Test
  ofstream log_file;
  log_file.open(result_path, std::fstream::out);
  test_net->ShareTrainedLayersWith(train_net.get());
  vector<Blob<double>* > bottom;
  for (int i = 0; i < TotalLength; ++i) { 
    double clip = i == 0 ? 0.0 : 1.0;
    test_data_layer->Reset(data.mutable_cpu_data(), &dummy, 1);
    test_clip_layer->Reset(&clip, &dummy, 1);
    const vector<Blob<double>* >& pred = test_net->Forward(bottom);
    CHECK_EQ(pred.size(), 1);
    CHECK_EQ(pred[0]->count(), 1);
    log_file << sequence.cpu_data()[i] << " " << *pred[0]->cpu_data() << endl;
  }
  
  log_file.close();
}
