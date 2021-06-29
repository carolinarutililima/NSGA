#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:09:58 2021

@author: carolinarutilidelima
"""
#import numpy as np
from ypstruct import structure
import random 
import math
import numpy as np

def run(problem, params):


    #Problem definition 
    func1 = problem.func1
    func2 = problem.func2
    varmin = problem.varmin
    varmax = problem.varmax
    ztd4 = problem.ztd4

    nvar = problem.nvar 
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma


    # Parameters 
    genmax = params.genmax
    npop = params.npop


    
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost_fc1 = None
    empty_individual.cost_fc2 = None
    
    # BestSolution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf 
    
    
    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(0, npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        if ztd4 == True:
            ZTD4f(pop[i])
       
        pop[i].cost_fc1 = func1(pop[i].position)
        pop[i].cost_fc2 = func2(pop[i].position)


    it=0
    
    while(it<genmax):
        function1_values = [func1(pop[i].position)for i in range(0,npop)]
        function2_values = [func2(pop[i].position)for i in range(0,npop)]
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
        
        print( "Interation {}".format(it))        

        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        new_pop = pop
        
        # creating offspring
        # mutation 
        #croossover
        #fitnees
        popc = [] 
        n_ch = npop//2
        for i in range(0,n_ch):
            # select parents RANDOM SELECTION
            q = np.random.permutation(npop)
            p1 = pop[q[0]]
            p2 = pop[q[1]] 
    
            
            # Perform Crossover           
            c1, c2 = crossover(p1,p2,gamma)
            
            # Perform Mutation 
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)
            
            #print("c2", c2.position[0])
            #print("c1",c1)


            
            # Check limits
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)



            if ztd4 == True:
                ZTD4f(c1)
                ZTD4f(c2)     


           # Evaluate First Offspring
            c1.cost_fc1 = func1(c1.position)
            c1.cost_fc2 = func2(c1.position)
                
            # Evaluate Second Offspring
            c2.cost_fc1 = func1(c2.position)
            c2.cost_fc2 = func2(c2.position)
                
            # Add Offsprings to population
            popc.append(c1) 
            popc.append(c2) 
            
        # Merge, sort and select
        pop = pop + popc


        function1_values2 = [func1(pop[i].position)for i in range(0,2*npop)]
        function2_values2 = [func2(pop[i].position)for i in range(0,2*npop)]
        
        
        
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if(len(new_solution)==npop):
                    break
            if (len(new_solution) == npop):
                break
        new_pop = [pop[i] for i in new_solution]
        pop = new_pop
        it = it + 1
            
       
    # Output     
    output = structure()
    output.fitness_func1 = function1_values
    output.fitness_func2 = function2_values
    output.pop = pop
    return output 
        
        

# Uniform crossover
def crossover(p1, p2, gamma):   
    
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(-gamma,1+gamma, c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)* p2.position
    c2.position = alpha*p2.position + (1-alpha)* p1.position
    
    return c1,c2

# Mutation
def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <=  mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma*np.random.randn(*ind.shape)
    
    return y

# apply bounds on x1
def ZTD4f(x):
    if x.position[0] > 1:
        x.position[0] = random.random()
    elif x.position[0] < 0:
        x.position[0] = random.random()
    return x

# Check limits    
def apply_bound(x, varmin, varmax): 

    x.position = np.clip(x.position, varmin, varmax)

    return x        

def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1


#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

 