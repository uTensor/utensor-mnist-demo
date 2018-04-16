# Basic MNIST handwriting recognition with uTensor
uTensor reduces AI-inference cost significantly, bringing AI to Cortex M devices.

This tutorial discusses at a high level what TensorFlow graphs are and how to start using uTensor to build a handwriting recognition application. This involves training a fully-connected neural network on the MNIST dataset on a host machine, generating embedded code, and building an mbed application that classifies handwritten digits based on user input.

## Introduction
TensorFlow is an open-source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, and the graph edges represent tensors, a type of multidimensional data array. 

uTensor translates these TensorFlow graphs into C++ code, which you can run on embedded devices.

Running artificial intelligence on embedded systems involves 3 main steps.

1. construct a machine learning model either offline on your local machine or in the cloud.
1. feed this model into code generation tools. These tools output uTensor kernels, which you can run on your devices. These kernels are functions that output predictions based on their inputs.
1. build embedded applications that call these functions to make informed decisions.

## Requirements
- DISCO_F413ZH
- SD Card (see SD card section)
- [mbed-cli](https://os.mbed.com/docs/v5.7/tools/installation-and-setup.html)
- [uTensor-cli](https://github.com/uTensor/utensor_cgen) `pip install utensor-cgen`
- [Tensorflow 1.6+](https://www.tensorflow.org/install)


## Build Instruction
1. Import the project:
```
mbed import https://github.com/uTensor/utensor-mnist-demo
```

### Train the model
The process of training and validating the model is exactly the same as in traditional machine learning workflows. This example trains a fully connected neural net with two hidden layers to recognize handwritten digits from the MNIST dataset, but you can apply the concepts to an application of your choice.

In preparation for code generation, you must freeze the TensorFlow model. Freezing a model stores learned graph parameters in a protobuf file.


- Train the tensorflow model
  1. go to `tensorflow-models/`
  2. run `python deep_mlp.py`
    - `python deep_mlp.py -h` for help
  3. training process should start immediately
  <img alt=train-mlp src=docs/images/train_mlp.png width=400/>
  4. the output pb file will be `mnist_model/deep_mlp.pb`
  5. go back to project root directory

### Generate embedded C++ code

You can get the output node names from the IPython notebook. This will create a `models` and `constants` directory. `models` contains the embedded code interface for making inferences in your applications, and `constants` contains the *learned* weights associated with each stage in the neural net.

```
utensor-cli --output-nodes=OutputLayer/y_pred tensorflow-models/mnist_model/deep_mlp.pb
```
### Prepare the mbed project
This example builds a handwriting recognition application using Mbed and the generated model, but you can apply these concepts to your own projects and platforms. This example uses the \texttt{ST-Discovery-F413H} because it has a touch screen and SD card built in, but you could just as easily build the application using plug-in components.

1. Copy the constants folder to your SD card, and insert SD card in board.
1. Run `mbed deploy`, this fetches the necessary libraries like uTensor
1. Build the mbed project:
```
mbed compile -m DISCO_F413ZH -t GCC_ARM --profile=uTensor/build_profile/release.json
```
1. Finally flash your device by dragging and dropping the binary from `BUILD/DISCO_F413ZH/GCC_ARM/utensor-mnist-demo.bin` to your device.

# Playing with the application
After drawing a number on the screen press the blue button to run inference, uTensor should output its prediction in the middle of the screen. Then press the reset button.  
[![Whoops! Failed loading video](https://img.youtube.com/vi/FhbCAd0sO1c/0.jpg)](https://www.youtube.com/watch?v=FhbCAd0sO1c)
