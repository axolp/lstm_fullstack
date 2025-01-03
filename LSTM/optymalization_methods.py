import numpy

def gradient_descent(wages, gradient, lr):
    return wages - (lr * gradient)

def gradient_descent_with_momentum(wages, gradient, previous_gradient, lr, momentum):
    v_t = momentum * previous_gradient + (1 - momentum) * gradient  
    return wages - lr * v_t  