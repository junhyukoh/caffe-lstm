#include <cuda_runtime.h>
#include <ctime>
#include "caffe/caffe.hpp"
#include <math.h>
#include <iostream>

using namespace caffe;
using namespace std;

const int SEQ_LENGTH = 320;

double f_x(double t) {
  return 0.5*sin(2*t) - 0.05*cos(17*t + 0.8) 
      + 0.05*sin(25*t+10) - 0.02*cos(45*t + 0.3);
}

int main(int argc, char** argv)
{
  if (argc < 3) {  
    LOG(ERROR) << "lstm_sequence solver_proto(1) result_path(2)";
    return 0;
  }

  //converting input parameters
  const char* net_solver(argv[1]);
  const char* result_path(argv[2]);

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(net_solver, &solver_param);

  shared_ptr<Solver<double> > solver;
  solver.reset(GetSolver<double>(solver_param));
  shared_ptr<Net<double> > train_net;
  shared_ptr<Net<double> > test_net;

  LOG(INFO) << "Starting Optimization";
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
  double mean = 0;
  double max_abs = 0;
  for (int i = 0; i < SEQ_LENGTH; ++i) {
    double val = f_x(i * 0.01);
    max_abs = max(max_abs, abs(val));
  }

  for (int i = 0; i < SEQ_LENGTH; ++i) {
    mean += f_x(i * 0.01) / max_abs;
  }

  mean /= SEQ_LENGTH;

  for (int i = 0; i < SEQ_LENGTH; ++i) {
    vector<double> l;
    double y = f_x(i*0.01) / max_abs - mean;
    l.push_back(y);
    data.push_back(datum);
    labels.push_back(l);
  }

  vector<double> losses;
  double smoothed_loss = 0;

  train_net = solver->net();
  while(!solver->IsFinished()) {
    const vector<shared_ptr<Layer<double> > >& layers = train_net->layers();
    ((SeqMemoryDataLayer<double>*)layers[0].get())->DataFetch(data, labels);
    solver->SolveIter(smoothed_loss, losses);
  }

  test_net = solver->test_nets()[0];
  test_net->ShareTrainedLayersWith(train_net.get());
  vector<Blob<double>* > bottom;
  const vector<Blob<double>* >& result = test_net->Forward(bottom);

  ofstream log_file;
  log_file.open(result_path, std::fstream::out);

  const double* output = result[0]->cpu_data();
  CHECK_EQ(result[0]->count(), SEQ_LENGTH);

  for (int i = 0; i < SEQ_LENGTH; ++i) {
    vector<double>& l = labels[i];
    log_file << l[0] << " " << output[i] << endl;
  }

  log_file.close();
}
