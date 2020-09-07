#!/usr/bin/env python
# coding: utf-8

# In[17]:


"""
Created on Tues Sep  1 13:45:06 2020

@Group 3: Youssaf & Rashedur

problem: Use the Poisson distribution as a candidate distribution to generate samples from the bino-
mial distribution
"""

# here first we will import the numpy package with random module 
from numpy import random 
#here we ill import matplotlib 
import matplotlib.pyplot as plt 
#now we will import seaborn 
import seaborn as sns  
#we will plot a displot here 
sns.distplot(random.binomial(n=200,p=0.05,size=1000), hist=False, label='binomial')
#we will plot a displot here 
sns.distplot(random.poisson(lam=10,size=1000), hist=False, label='poisson')  
# now we have the plot printed 
plt.show()

#size=2000 for 5/A
#size=1000 for 3/A 


# In[18]:


"""
Created on Tues Sep  1 13:45:06 2020

 

@Group 3: Youssaf & Rashedur

 

problem: Use the Poisson distribution as a candidate distribution to generate samples from the bino-
mial distribution

 

Prof: Josic & Prof:Stewart

 

"""

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()

# f is the target function : binomial
# g is the candidate function: poisson


#parameters for binomial
n=500
p=0.05
#n*p = 25
#parameter for poisson
lamda=[15,20,25,35]

# Rejection Sampling
X = []

c = 1.5
print("c1= ", c) 

iteration=200
for i in range(len(lamda)):
    
    while(len(X) < iteration):
        
        X1=np.random.poisson(lamda[i])   #simulating X variable from poisson
        U=np.random.uniform(0,1)         #uniform
        f=sps.binom.pmf(X1, n, p)      
        g=sps.poisson.pmf(X1, lamda[i])   
        
        if(U <(f/(c*g))):
            X.append(X1)
            
        c= max(c, f/g) 
       
    plt.title("Rejection Sampling Dist for $\lambda$ = " + str(lamda[i]))
    
    
    B = np.random.binomial(n,p,iteration)
    
    plt.title("Dist for $\lambda$ = " + str(lamda[i]))
    
    
    sns.distplot(X, hist=False, label= "X")
    sns.distplot(B, hist=False, label= "B")

    plt.show()
    print("acceptance rate= ", 100/c)


# In[19]:


"""
Created on Tues Sep  1 13:45:06 2020

 

@Group 3: Youssaf & Rashedur

 

problem: Use the Poisson distribution as a candidate distribution to generate samples from the bino-
mial distribution

 

Prof: Josic & Prof: Stewart

 

"""
        
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
import time



sns.set()

# f is the target function : binomial
# g is the candidate function: poisson
# The candidate g has heavier tails for lamda ~ n*p

#parameters for binomial
n=500 # n>100
p=0.05 # p<<1 very small
#parameters for poisson around n*p=25
lamda=[15,20,25,35]

# Rejection Sampling
X = []

c = 5 # 5,10, 15 

print("c1= ", c)


iteration=200 # generate iteration smaples from the candidate distribution. 



for i in range(len(lamda)):
    while(len(X) < iteration):
        X1=np.random.poisson(lamda[i])   #simulating X variable from poisson
        U=np.random.uniform(0,1)         #uniform
        f=sps.binom.pmf(X1, n, p)      
        g=sps.poisson.pmf(X1, lamda[i])     
            
        if(U <(f/(c*g))):
            X.append(X1)  # accept X1 ~ g 
        c= max(c, f/g) 
    plt.title("Rejection Sampling Dist for $\lambda$ = " + str(lamda[i]))
        
    F = np.random.binomial(n,p,iteration)
#   G = np.random.binomial(n,p,iteration)
    plt.title("Dist for $\lambda$ = " + str(lamda[i]))
    
    
    sns.distplot(X, hist=False, label= "X: Samples")
    sns.distplot(F, hist=False, label= "F: target")

    plt.show()
    print("c= ", c)
    print("Aceptance rate= ", 100/c)


# In[20]:


"""
Created on Tues Sep  1 13:45:06 2020

 

@Group 3: Youssaf & Rashedur

 

problem: Use the Poisson distribution as a candidate distribution to generate samples from the bino-
mial distribution

 

Prof: Josic & Prof: Stewart

 

"""
        
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
import time



sns.set()

# f is the target function : binomial
# g is the candidate function: poisson
# The candidate g has heavier tails for lamda ~ n*p

#parameters for binomial
n=500 # n>100
p=0.05 # p<<1 very small
#parameters for poisson around n*p=25
lamda=[15,20,25,35]

# Rejection Sampling
X = []

c = 10 # 5,10, 15 

print("c1= ", c)


iteration=200 # generate iteration smaples from the candidate distribution. 



for i in range(len(lamda)):
    while(len(X) < iteration):
        X1=np.random.poisson(lamda[i])   #simulating X variable from poisson
        U=np.random.uniform(0,1)         #uniform
        f=sps.binom.pmf(X1, n, p)      
        g=sps.poisson.pmf(X1, lamda[i])     
            
        if(U <(f/(c*g))):
            X.append(X1)  # accept X1 ~ g 
        c= max(c, f/g) 
    plt.title("Rejection Sampling Dist for $\lambda$ = " + str(lamda[i]))
        
    F = np.random.binomial(n,p,iteration)
#   G = np.random.binomial(n,p,iteration)
    plt.title("Dist for $\lambda$ = " + str(lamda[i]))
    
    
    sns.distplot(X, hist=False, label= "X: Samples")
    sns.distplot(F, hist=False, label= "F: target")

    plt.show()
    print("c= ", c)
    print("Aceptance rate= ", 100/c)


# In[21]:


"""
Created on Tues Sep  1 13:45:06 2020

 

@Group 3: Youssaf & Rashedur

 

problem: Use the Poisson distribution as a candidate distribution to generate samples from the bino-
mial distribution

 

Prof: Josic & Prof: Stewart

 

"""
        
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
import time



sns.set()

# f is the target function : binomial
# g is the candidate function: poisson
# The candidate g has heavier tails for lamda ~ n*p

#parameters for binomial
n=500 # n>100
p=0.05 # p<<1 very small
#parameters for poisson around n*p=25
lamda=[15,20,25,35]

# Rejection Sampling
X = []

c = 15 # 5,10, 15 

print("c1= ", c)


iteration=200 # generate iteration smaples from the candidate distribution. 



for i in range(len(lamda)):
    while(len(X) < iteration):
        X1=np.random.poisson(lamda[i])   #simulating X variable from poisson
        U=np.random.uniform(0,1)         #uniform
        f=sps.binom.pmf(X1, n, p)      
        g=sps.poisson.pmf(X1, lamda[i])     
            
        if(U <(f/(c*g))):
            X.append(X1)  # accept X1 ~ g 
        c= max(c, f/g) 
    plt.title("Rejection Sampling Dist for $\lambda$ = " + str(lamda[i]))
        
    F = np.random.binomial(n,p,iteration)
#   G = np.random.binomial(n,p,iteration)
    plt.title("Dist for $\lambda$ = " + str(lamda[i]))
    
    
    sns.distplot(X, hist=False, label= "X: Samples")
    sns.distplot(F, hist=False, label= "F: target")

    plt.show()
    print("c= ", c)
    print("Aceptance rate= ", 100/c)


# In[ ]:




