# LSTM Implementation in [Caffe](http://caffe.berkeleyvision.org)
 Note that the master branch of Caffe supports LSTM now. ([Jeff Donahue's implementation](https://github.com/BVLC/caffe/pull/2033) has been merged.) <br />
 This repo is no longer maintained. <br />
 
## Speed comparison (Titan X, 3-layer LSTM with 2048 units)
 Jeff's code is more modularized, whereas this code is optimized for LSTM. <br />
 This code computes gradient w.r.t. recurrent weights with a single matrix computation. <br />

  * Batch size = 20, Length = 100
  
  | Code           | Forward(ms) | Backward(ms)  | Total (ms) |
  | -------------- |-------------|---------------|------------|
  | **This code**  | **248**     | **291**       | **539**    |
  | Jeff's code    | 264         | 462           | 726        |

  * Batch size = 4, Length = 100
  
  | Code           | Forward(ms) | Backward(ms)  | Total (ms) |
  | -------------- |-------------|---------------|------------|
  | **This code**  | **131**     | **118**       | **249**    |
  | Jeff's code    | 140         | 290           | 430        |

  * Batch size = 20, Length = 20
  
  | Code           | Forward(ms) | Backward(ms)  | Total (ms) |
  | -------------- |-------------|---------------|------------|
  | **This code**  | **49**      | **59**        | **108**    |
  | Jeff's code    | 52          | 92            | 144        |

  * Batch size = 4, Length = 20
  
  | Code           | Forward(ms) | Backward(ms)  | Total (ms) |
  | -------------- |-------------|---------------|------------|
  | **This code**  | **29**      | **26**        | **55**     |
  | Jeff's code    | 30          | 61            | 91         |


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
