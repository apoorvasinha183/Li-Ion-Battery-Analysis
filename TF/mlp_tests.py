# %% imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# %% Vint 
F = 96487.0
V_INT_k = lambda x,i: (2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i)

V_INT = lambda x,A: np.dot(A, np.array([V_INT_k(x,i) for i in range(len(A))])) / F

def Ai(A,i,a):
    A[i]=a
    return A

Ap = np.array([
    -31593.7,
    0.106747,
    24606.4,
    -78561.9,
    13317.9,
    307387.0,
    84916.1,
    -1.07469e+06,
    2285.04,
    990894.0,
    283920,
    -161513,
    -469218
])


X = np.linspace(0.0,1.0,100)
I = np.ones(100)
Vint = np.array([V_INT(x,Ap) for x in X])

#plt.plot(X,Vint)


mlp = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(1),
])
mlp.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse", metrics=["mae"])
mlp.summary()

Y = np.hstack([np.linspace(0.85,-0.2,90), np.linspace(-0.25,-0.8,10)])

def scheduler(epoch):
    if epoch < 800:
        return 2e-2
    elif epoch < 1100:
        return 1e-2
    elif epoch < 2200:
        return 5e-3
    else:
        return 1e-3

mlp.fit(X,Y, epochs=2400, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])


print(mlp.get_weights())
b = np.array(mlp.get_weights(),dtype='object')
print("Shape of this array is ",b.shape)

try:
    np.save('../training/mlp_initial_weight_with-I.npy', b)
except:
    file = open('../training/mlp_initial_weight_with-I.npy', 'w+')   
    np.save('../training/mlp_initial_weight_with-I.npy', mlp.get_weights()) 

pred = mlp.predict(X)


plt.plot(X,Y, color='gray')
plt.plot(X,pred)
plt.grid()

plt.show()

