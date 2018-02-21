# MNIST
Handwritten digit recognition demo

# Requirements
- DISCO_F413ZH
- SD Card (see SD card section)
- [mbed-cli](https://os.mbed.com/docs/v5.7/tools/installation-and-setup.html)
- [uTensor-cli](https://github.com/uTensor/utensor_cgen) `pip install utensor-cgen`

# Build Instruction
1. Import the project:
```
mbed import https://github.com/uTensor/utensor-mnist-demo
```
1. Train the tensorflow model
    1. Launch IPython `jupyter notebook` ![Launch the notebook](https://github.com/uTensor/utensor-mnist-demo/blob/quickstart/docs/images/jupyter.png)
    1. Open the `tensorflow-models/deep_mlp.ipynb`
    1. Select `Kernel/Restart & Run All`. This will build a 2 layer NN then train, quantize, and saves the model in `tensorflow-models/my-model/deep_mlp.pb`.
    ![Run the IPython notebook](https://github.com/uTensor/utensor-mnist-demo/blob/quickstart/docs/images/kernel.png)
1. Generate CPP code. You can get the output node names from the IPython notebook. This will create a `models` and `constants` directory. `models` contains the CPP code and `constants` contains the weights of the graph.
```
utensor-cli --output-nodes Prediction/y_pred tensorflow-models/my-model/deep_mlp.pb
```
1. Copy the constants folder to your SD card, and insert SD card in board.
1. Run `mbed deploy`, this fetches the necessary libraries like uTensor
1. Build the mbed project:
```
mbed compile -m DISCO_F413ZH -t GCC_ARM --profile=utensor/build_profile/release.json -f
```
1. Finally flash your device by dragging and dropping the binary from `BUILD/DISCO_F413ZH/GCC_ARM/utensor-mnist-demo.bin` to your device.

# Playing with the application
After drawing a number on the screen press the blue button to run inference, uTensor should output its prediction in the middle of the screen. Then press the reset button.  
