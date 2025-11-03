import math
import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def chain_rule():
    """
    Compute df/dz, df/dq, df/dx, and df/dy for f(x,y,z)=xy+z,
    where q=xy, at x=-2, y=5, z=-4.
    Return them in this order: df/dz, df/dq, df/dx, df/dy. 
    """ 
    df_dz = 1.0
    df_dq = 1.0
    df_dx = 5.0 # y
    df_dy = -2.0 # x
    return df_dz, df_dq, df_dx, df_dy

def ReLU():
    """
    Compute dx and dw, and return them in order.
    Forward:
        y = ReLU(w0 * x0 + w1 * x1 + w2)

    Returns:
        dx -- gradient with respect to input x, as a vector [dx0, dx1]
        dw -- gradient with respect to weights (including the third term w2), 
              as a vector [dw0, dw1, dw2]
    """

    x = [-1, -2]
    w = [2, -3, -3]
    
    # Forward pass
    z = w[0] * x[0] + w[1] * x[1] + w[2]  # z = 1
    y = max(0, z)  # y = 1, ReLU(1) = 1
    
    # Backward pass
    dy_dz = 1 if y > 0 else 0  # = 1
    
    # Gradients w.r.t. weights
    dw = [
        dy_dz * x[0],      # dw0 = 1 * (-1) = -1
        dy_dz * x[1],      # dw1 = 1 * (-2) = -2
        dy_dz * 1          # dw2 = 1 * 1 = 1
    ]
    
    # Gradients w.r.t. inputs
    dx = [
        dy_dz * w[0],      # dx0 = 1 * 2 = 2
        dy_dz * w[1]       # dx1 = 1 * (-3) = -3
    ]
    
    return dx, dw

import torch

def chain_rule_a():
    """
    In the lecture notes, the last three forward pass values are
    a=0.37, b=1.37, and c=0.73.
    Calculate these numbers to 4 decimal digits and return in order of a, b, c
    """
    # From lec05b
    w0 = 2.00
    x0 = -1.00
    w1 = -3.00
    x1 = -2.00
    w2 = -3.00

    # Forward
    s3 = w0 * x0 + w1 * x1 + w2 
    s4 = -s3

    a = math.exp(s4)
    b = a + 1
    c = 1 / b

    # Round to 4 decimal digits
    a_val = round(a, 4)
    b_val = round(b, 4)
    c_val = round(c, 4)

    # print(f"a: {a_val}, b: {b_val}, c: {c_val}")

    return a_val, b_val, c_val


import torch

def chain_rule_b():
    """
    In the lecture notes, the backward pass values are 0.20, 0.39, -0.59, and -0.53.
    Calculate these numbers to 4 decimal digits and return gradients in order of
    w0, x0 , w1, x1, w2.
    """
    # Initialize variables as tensors with requires_grad=True
    w0 = torch.tensor(2.00, requires_grad=True)
    x0 = torch.tensor(-1.00, requires_grad=True)
    w1 = torch.tensor(-3.00, requires_grad=True)
    x1 = torch.tensor(-2.00, requires_grad=True)
    w2 = torch.tensor(-3.00, requires_grad=True)

    # Forward pass
    s3 = w0 * x0 + w1 * x1 + w2
    s4 = -s3
    a = torch.exp(s4)
    b = a + 1
    c = 1 / b

    # Backward pass
    c.backward()

    # Get gradients from .grad attribute and round to 4 decimal digits
    gw0 = round(w0.grad.item(), 4)
    gx0 = round(x0.grad.item(), 4)
    gw1 = round(w1.grad.item(), 4)
    gx1 = round(x1.grad.item(), 4)
    gw2 = round(w2.grad.item(), 4)

    print(f"gw0: {gw0}, gx0: {gx0}, gw1: {gw1}, gx1: {gx1}, gw2: {gw2}")

    return torch.tensor([gw0, gx0, gw1, gx1, gw2])


def backprop_a():
    """
    Let f(w,x) = torch.tanh(w0x0+w1x1+w2).  
    Assume the weight vector is w = [w0=5, w1=2], 
    the input vector is  x = [x0=-1,x1= 4],, and the bias is  w2  =-2.
    Use PyTorch to calculate the forward pass of the network, return y_hat = f(w,x).
    """

    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)

    # Forward pass
    z = w0 * x0 + w1 * x1 + w2
    y_hat = torch.tanh(z)
    
    return y_hat

def backprop_b():
    """
    Use PyTorch Autograd to calculate the gradients 
    for each of the weights, and return the gradient of them 
    in order of w0, w1, and w2.
    """

    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)

    # Forward pass
    z = w0 * x0 + w1 * x1 + w2
    y_hat = torch.tanh(z)

    # MSE Loss
    y_true = torch.tensor(1.0)
    loss = (y_hat - y_true) ** 2

    # Backward pass
    loss.backward()
    
    return w0.grad, w1.grad, w2.grad

def backprop_c():
    """
    Assuming a learning rate of 0.1, 
    update each of the weights accordingly. 
    For simplicity, just do one iteration. 
    And return the updated weights in the order of w0, w1, and w2 
    """
    
    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)

    # Forward pass
    z = w0 * x0 + w1 * x1 + w2
    y_hat = torch.tanh(z)

    # MSE Loss
    y_true = torch.tensor(1.0)
    loss = (y_hat - y_true) ** 2

    # Backward pass
    loss.backward()

    # Learning rate
    lr = 0.1

    # Weight update
    with torch.no_grad():
        w0_new = w0 - lr * w0.grad
        w1_new = w1 - lr * w1.grad
        w2_new = w2 - lr * w2.grad

    return w0_new, w1_new, w2_new


def constructParaboloid(w=256, h=256):
    img = np.zeros((w, h), np.float32)
    for x in range(w):
        for y in range(h):
            # let's center the paraboloid in the img
            img[y, x] = (x - w / 2) ** 2 + (y - h / 2) ** 2
    return img


def newtonMethod(x0, y0):
    # paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.from_numpy(constructParaboloid())
    paraboloid = torch.unsqueeze(paraboloid, 0) 
    paraboloid = torch.unsqueeze(paraboloid, 0)    # -> (1,1,H,W) for conv2d
    
    x, y = float(x0), float(y0)

    tolerance = 1e-4
    iterations = 50
    
    for iter_count in range(iterations):  
        # Current position
        xi, yi = int(round(x)), int(round(y))

        # Boundary check
        if xi < 1 or xi >= paraboloid.shape[3]-1 or yi < 1 or yi >= paraboloid.shape[2]-1:
            break
        
        # Calculate gradient
        fx = (paraboloid[0, 0, yi, xi+1] - paraboloid[0, 0, yi, xi-1]) / 2.0
        fy = (paraboloid[0, 0, yi+1, xi] - paraboloid[0, 0, yi-1, xi]) / 2.0
        
        # Calculate Hessian
        fxx = paraboloid[0, 0, yi, xi+1] - 2*paraboloid[0, 0, yi, xi] + paraboloid[0, 0, yi, xi-1]
        fyy = paraboloid[0, 0, yi+1, xi] - 2*paraboloid[0, 0, yi, xi] + paraboloid[0, 0, yi-1, xi]
        fxy = (paraboloid[0, 0, yi+1, xi+1] - paraboloid[0, 0, yi+1, xi-1] - 
               paraboloid[0, 0, yi-1, xi+1] + paraboloid[0, 0, yi-1, xi-1]) / 4.0
        
        # Newton method: [x, y] = [x, y] - H^{-1} * grad
        H = torch.tensor([[fxx, fxy], [fxy, fyy]], dtype=torch.float32)
        grad = torch.tensor([fx, fy], dtype=torch.float32)
        
        try:
            H_inv = torch.inverse(H)
            delta = torch.matmul(H_inv, grad)
            x -= delta[0].item()
            y -= delta[1].item()
        except:
            break
        
        # Convergence check
        if torch.norm(delta) < tolerance:
            iter_count += 1
            break

    print(f"Converged to ({x}, {y}) in {iter_count} iterations")
    
    return int(round(x)), int(round(y)), iter_count


def sgd(x0, y0, lr=0.001):
    # paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.from_numpy(constructParaboloid())
    paraboloid = torch.unsqueeze(paraboloid, 0)
    paraboloid = torch.unsqueeze(paraboloid, 0)

    x, y = float(x0), float(y0)

    tolerance = 1e-4
    iterations = 5000
    
    for epoch in range(iterations):
        xi, yi = int(round(x)), int(round(y))
        
        # Boundary check
        if xi < 1 or xi >= paraboloid.shape[3]-1 or yi < 1 or yi >= paraboloid.shape[2]-1:
            break
        
        # Calculate gradients
        fx = (paraboloid[0, 0, yi, xi+1] - paraboloid[0, 0, yi, xi-1]) / 2.0
        fy = (paraboloid[0, 0, yi+1, xi] - paraboloid[0, 0, yi-1, xi]) / 2.0
        
        # SGD update
        x -= lr * fx.item()
        y -= lr * fy.item()
        
        # Convergence check
        if abs(fx.item()) < tolerance and abs(fy.item()) < tolerance:
            break
    
    print(f"Converged to ({x}, {y}) in {epoch + 1} iterations")
    
    return int(round(x)), int(round(y)), epoch + 1


if __name__ == "__main__":
    # chain_rule()
    # ReLU()
    # chain_rule_a()
    # chain_rule_b()
    # backprop_a()
    # backprop_b()
    # backprop_c()
    # newtonMethod(200, 50)
    sgd(200, 50, lr=0.001)