import numpy as np
from time import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
DTYPE = 'float64'

#from BatteryRNNCell import BatteryRNNCell
from BatteryRNNCell_mlp import BatteryRNNCell,BatteryRNN
from BatteryRNNCell_PINN import BatteryRNNCell_PINN, BatteryRNN_PINN

def get_model(return_sequences=True, stateful=False, dtype=DTYPE, dt=1.0, mlp=True, mlp_trainable=True, share_q_r=True, q_max_base=None, R_0_base=None, D_trainable=False,WARM_START=True, version = "original_paper"):
    if version == "original_paper":
        model = BatteryRNN(mlp_trainable=mlp_trainable,dt=dt,q_max_base=q_max_base,R_0_base=R_0_base,D_trainable=D_trainable,WARM_START= WARM_START)
    if version == "naive_PINN":
        model = BatteryRNN_PINN(mlp_trainable=mlp_trainable,dt=dt,q_max_base=q_max_base,R_0_base=R_0_base,D_trainable=D_trainable,WARM_START= WARM_START)
    return model

if __name__ == "__main__":
    # test keral model pred
    #TODO: Replicate this in pytorch
    inputs = np.ones((700,1000), dtype=DTYPE) * np.linspace(1.0,2.0,1000)  # constant load
    inputs = inputs.T[:,:,np.newaxis]

    model = get_model(dt=10.0, mlp=True)
    #model = get_model(batch_input_shape=inputs.shape, dt=10.0, mlp=False)
    model.summary()

    start = time()
    pred = model.predict(inputs)
    duration = time() - start
    print("Inf. time: {:.2f} s - {:.3f} ms/step ".format(duration, duration/inputs.shape[1]*1000))

    cmap = matplotlib.cm.get_cmap('Spectral')

    fig = plt.figure()

    plt.subplot(211)
    for i in range(inputs.shape[0]):
        plt.plot(inputs[i,:,0], color=cmap(i/1000))
    plt.ylabel('I (A)')
    plt.grid()

    plt.subplot(212)
    for i in range(pred.shape[0]):
        plt.plot(pred[i,:], color=cmap(i/1000))
    plt.ylabel('Vm (V)')
    plt.grid()

    plt.xlabel('Time (s)')

    plt.show()