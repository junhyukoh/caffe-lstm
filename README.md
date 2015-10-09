# LSTM Implementation in [Caffe](http://caffe.berkeleyvision.org)
 Note that [Jeff Donahue's implementation](https://github.com/BVLC/caffe/pull/2033) will be merged to Caffe (not this code).
  * Jeff's code is more modularized, while this code is optimized for LSTM.
  * This code computes gradient w.r.t. recurrent weights in a single matrix computation.
  * Speed comparison (GTX Titan X, 3-layer LSTM with 2048 units, batch size of 20)

  | Code           | Forward(ms) | Backward(ms)  | Total (ms) |
  | -------------- |-------------|---------------|------------|
  | **This code**  | **248**     | **291**       | **539**    |
  | Jeff's code    | 264         | 462           | 726        |

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
