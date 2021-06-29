

import matplotlib.pyplot as plt
from ypstruct import structure
import math
import NSGAv4 


#First function to optimize
def function1(x):
    result = - x[0]
    return result

#Second function to optimize
def functiong(x):
    b = 0
    for i in x[1:]:
        a = i** 2 - 10*math.cos(4*math.pi*i)
        b = a + b
    result =  1 + 10*(len(x)-1) + b
    return result 

def function2(x):
    fung = functiong(x)
    result = - fung * (1- math.sqrt(x[0]/fung))     
    return result



# Variables according to problem definition
problem = structure()
problem.func1 = function1
problem.func2 = function2 

problem.nvar = 10 
problem.varmin = -5
problem.varmax = 5
problem.ztd4 = True
# x1 - [0,1]



params = structure()
params.genmax = 1000 # max iterations number
params.npop = 30 # population number
params.gamma = 0.1  # crossover parameter
params.sigma = 0.3 # mutation parameter 
params.mu = 0.03  # mutation % genes
#params.beta = 2 # selection for the rolette wheel
#params.pc = 1 # variation initial population



output = NSGAv4.run(problem, params)

print(output.pop)


#Lets plot the final front now
function1 = [i * -1 for i in output.fitness_func1]
function2 = [j * -1 for j in output.fitness_func2]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()