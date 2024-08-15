import numpy as np

X = [0.5, 1.25, 2, 2.5, 3.75, 4.2]
Y = [0.2, 0.45, 0.65, 0.8, 0.95, 1]

def f(w,b,x): #sigmoid logistic function
    return 1.0/(1.0 + np.exp(-(w*x + b)))

def error(w,b): #loss fun
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5 * (fx - y) **2
    return err

def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx * (1 - fx) * x

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx * (1 - fx)

def do_gradient_descent():
    w, b, eta, max_epoch = -2, -2, 1, 100
    for i in range(max_epoch):
        dw, db = 0, 0
        for x,y in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        w = w - eta * dw
        b = b - eta * db
        print(w,b)
    print("Final loss is :",str(error(w,b)))
        
do_gradient_descent()