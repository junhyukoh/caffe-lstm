# LSTM Implementation in [Caffe](http://caffe.berkeleyvision.org)
  * This is my personal implementation of LSTM in Caffe with minimal modifications. <br />
  * The official Caffe is going to support LSTM/RNN (not my code). <br />
  * See the following link for the details [LSTM Pull Request](https://github.com/BVLC/caffe/pull/2033) <br />

# Example
An example code is in /examples/lstm_sequence/. <br />
In this code, LSTM network is trained to generate a predefined sequence without any inputs. <br />
This experiment was introduced by [Clockwork RNN](http://jmlr.org/proceedings/papers/v32/koutnik14.pdf). <br />
Four different LSTM networks and shell scripts(.sh) for training are provided. <br />
Each script generates a log file containing the predicted sequence and the true sequence. <br />
You can use plot_result.m to visualize the result. <br />
The result of four LSTM networks will be as follows:
  * 1-layer LSTM with 15 hidden units for short sequence
![Diagram](https://raw.githubusercontent.com/junhyukoh/caffe-lstm/master/examples/lstm_sequence/lstm-320-b320-h15.png)
  * 1-layer LSTM with 50 hidden units for long sequence
![Diagram](https://raw.githubusercontent.com/junhyukoh/caffe-lstm/master/examples/lstm_sequence/lstm-960-b320-h50.png)
  * 3-layer deep LSTM with 7 hidden units for short sequence
![Diagram](https://raw.githubusercontent.com/junhyukoh/caffe-lstm/master/examples/lstm_sequence/deep-lstm-320-b320-h7.png)
  * 3-layer deep LSTM with 23 hidden units for long sequence
![Diagram](https://raw.githubusercontent.com/junhyukoh/caffe-lstm/master/examples/lstm_sequence/deep-lstm-960-b320-h23.png)
