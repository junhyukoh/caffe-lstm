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
  const int SeqLength = atoi(argv[3]);

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(net_solver, &solver_param);

  shared_ptr<Solver<double> > solver;
  solver.reset(GetSolver<double>(solver_param));
  shared_ptr<Net<double> > train_net(solver->net());
  shared_ptr<Net<double> > test_net(solver->test_nets()[0]);

  solver->PreSolve();
  // Set device id and mode
  if (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    LOG(INFO) << "Use GPU with device ID " << solver_param.device_id();
    Caffe::SetDevice(solver_param.device_id());
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  const unsigned char input = 0;
  Datum datum;
  datum.set_channels(1);
  datum.set_width(1);
  datum.set_height(1);
  datum.set_data(&input, 1);

  vector<Datum> data;
  vector<vector<double> > labels;

  // Scale data to lie on [-1, 1]
  double mean = 0;
  double max_abs = 0;
  for (int i = 0; i < SeqLength; ++i) {
    double val = f_x(i * 0.01);
    max_abs = max(max_abs, abs(val));
  }

  // Subtract mean
  for (int i = 0; i < SeqLength; ++i) {
    mean += f_x(i * 0.01) / max_abs;
  }
  mean /= SeqLength;

  // Make t
  for (int i = 0; i < SeqLength; ++i) {
    vector<double> l;
    double y = f_x(i*0.01) / max_abs - mean;
    l.push_back(y);
    data.push_back(datum);
    labels.push_back(l);
  }

  // Make mini-batches
  const vector<shared_ptr<Layer<double> > >& layers = train_net->layers();
  const LayerParameter& layer_param = layers[0]->layer_param();
  int batchsize = layer_param.memory_data_param().batch_size();
  CHECK_EQ(SeqLength % batchsize, 0) << "sequence length should be"
    "divided by batchsize";
  
  vector<vector<Datum> > batch_data;
  vector<vector<vector<double> > > batch_labels;

  for (int i = 0; i < SeqLength / batchsize; ++i) {
    vector<Datum> d;
    vector<vector<double> > l;

    for (int j = 0; j < batchsize; ++j) {
      d.push_back(datum);
      int idx = i * batchsize + j;
      vector<double> tmp;
      tmp.push_back(labels[idx].at(0));
      l.push_back(tmp);
    }

    batch_data.push_back(d);
    batch_labels.push_back(l);
  }


  // Training
  vector<double> losses;
  double smoothed_loss = 0;

  Caffe::set_phase(Caffe::TRAIN);
  int iter = 0;
  while(!solver->IsFinished()) {
    int batch_idx = iter % (SeqLength / batchsize);

    vector<Datum>& batch_d = batch_data[batch_idx];
    vector<vector< double> >& batch_l = batch_labels[batch_idx];

    ((SeqMemoryDataLayer<double>*)layers[0].get())->DataFetch(
      batch_d, batch_l, batch_idx == 0);
    solver->SolveIter(smoothed_loss, losses);
    iter++;
  }

  // Output Test
  Caffe::set_phase(Caffe::TEST);
  ofstream log_file;
  log_file.open(result_path, std::fstream::out);
  test_net->ShareTrainedLayersWith(train_net.get());
  vector<Blob<double>* > bottom;
  const vector<shared_ptr<Layer<double> > >& test_layers = test_net->layers();
  for (int i = 0; i < SeqLength; ++i) { 
    ((SeqMemoryDataLayer<double>*)test_layers[0].get())->DataFetch(datum, i == 0);
    const vector<Blob<double>* >& result = test_net->Forward(bottom);
    CHECK_EQ(result.size(), 1);
    const double* output = result[0]->cpu_data();
    CHECK_EQ(result[0]->count(), 1);
    vector<double>& l = labels[i];
    log_file << l[0] << " " << output[0] << endl;
  }
  
  log_file.close();
}
