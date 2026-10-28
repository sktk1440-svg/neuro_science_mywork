import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        temp = x[idx]
        x[idx] = temp + h
        fxh1 = f(x)
        x[idx] = temp-h
        fxh2 = f(x)
        x[idx] = temp
        grad[idx] = (fxh1 - fxh2) / (2*h)

    return grad

def gradient_decent(f,init_x,lr=0.1,step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
        print(x)
    return x

init_x = np.array([-3.0,4.0])
print(gradient_decent(function_2,init_x=init_x,lr=0.1,step_num=100))