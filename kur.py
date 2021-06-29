#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:48:28 2021

@author: carolinarutilidelima
"""
#import numpy as np 
import matplotlib.pyplot as plt
from ypstruct import structure
import math
import NSGAv4 



def f1(x):
    s = 0
    for i in range(len(x)-1):
        s += -10*math.exp(-0.2*math.sqrt(x[i]**2 + x[i+1]**2))
    return s

def f2(x):
    s = 0
    for i in range(len(x)):
        s += abs(x[i])**0.8 + 5*math.sin(x[i]**3)
    return s

# Variables according to problem definition
problem = structure()
problem.func1 = f1
problem.func2 = f2 

problem.nvar = 3 
problem.varmin = - 5 
problem.varmax = 5



params = structure()
params.max_gen = 2000 # max iterations number
params.npop = 20 # population number
params.pc = 1 # variation initial population
params.gamma = 0.1  # crossover parameter
params.sigma = 0.2 # mutation parameter 
params.mu = 0.02  # mutation % genes
params.beta = 2 # selection for the rolette wheel



output = NSGAv4.run(problem, params)



#Lets plot the final front now
function1 = [i * -1 for i in output.fitness_func1]
function2 = [j * -1 for j in output.fitness_func2]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()