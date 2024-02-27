import torch
#import pandas as pd
#import numpy as np
#import turtle
#import matplotlib.pyplot as plt
print(f"\n Verze Pytorche je: {torch.__version__}")

# Set manual seed
#torch.manual_seed(0) # cili vzdy stejna nahodna cisla .......

# Set random seed on the GPU
torch.cuda.manual_seed(1234)

def tisk(A, B):
    print(f"\n {B} je:\n {A} \n o rozmeru: {A.shape} \n zarizeni je {A.device}")

# Create random tensor
X = torch.rand(size=(7, 7)).to("cuda")
#print(f"X je:\n {X} \n o rozmeru: {X.shape} \n zarizeni je {X.device}")
tisk(X, "X")

# Create another random tensor
Y = torch.rand(size=(1, 7)).to("cuda")
tisk(Y, "Y")

Z = torch.matmul(Y, X) # je ok !!!!
tisk(Z, "Z")
Z = torch.matmul(X, Y.T) # no error because of transpose; jiny vysledek ovsem...
tisk(Z, "Z")

# Remove single dimensions
S = Z.squeeze()
tisk(S, "S")

# Find max
max = torch.max(S)

# Find min
min = torch.min(S)
print(max, min)

# Find arg max
arg_max = torch.argmax(S)

# Find arg min
arg_min = torch.argmin(S)
print(arg_max, arg_min)



