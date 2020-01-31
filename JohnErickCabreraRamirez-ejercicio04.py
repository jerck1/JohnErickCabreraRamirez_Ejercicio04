#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
#estandarizacion:
import sklearn.preprocessing
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Lectura de datos

data = pd.read_csv('Cars93.csv')


# In[4]:


# Selección de target y predictores


# In[6]:


Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.ones((93,12))
X[:,:11] = np.array(data[columns])
X[:,11] = np.array(data['Price'])


# In[7]:


# Renormalización de los datos para que todas las variables sean comparables


# In[12]:


X_scaled = np.ones((93,12))
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X[:,:11])
X_scaled[:,:11] = scaler.transform(X[:,:11])
X_scaled[:,11] = X[:,11] 


# In[13]:


regresion = sklearn.linear_model.LinearRegression()


# In[14]:


def compute_betas(X, Y):
    n_points = len(Y)
    # esta es la clave del bootstrapping: la seleccion de indices de "estudiantes"
    indices = np.random.choice(np.arange(n_points), n_points)
    new_X = X[indices, :]
    new_Y = Y[indices]
    regresion = sklearn.linear_model.LinearRegression()
    regresion.fit(new_X, new_Y)
    return regresion.coef_


# In[15]:


compute_betas(X_scaled, Y)


# In[71]:


n=len(columns)
#x_da=[]
a=[0,1,2,3,4,5,6,7,8,9,10]
beta=np.zeros((n,1))
#x_ran=X_scaled
#y_ran=np.ones(20)
for i in range(1,11):
    index=list(itertools.combinations(a,i))
    x_ran[:,i]=data[columns[index[i][0]]]
#    x_ran[:,i]=np.random.choice(data[columns[index[i][0]]],20)
    y_ran=np.random.choice(Y,20)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x_ran, y_ran, test_size=1)
    regresion.fit(data[columns[index[i][0]]], Y)
#regresion.coef_ esta en el mismo orden que columnas
#    beta[i][0]=regresion.intercept_
#    beta[i][0]=regresion.coef_
    print(regresion.coef_)
    print(regresion.score(X, Y))


# In[66]:


plt.scatter(beta, a)


# In[59]:


a=list(itertools.combinations(index,2))
print(a[0][0])
print(index[2][0])
#print(columns[index[0][0]])
data[columns[index[2][0]]]
print(x_da)


# In[8]:


# Un primer ajuste lineal con mínimos cuadrados



# In[64]:


np.zeros((10,1))


# In[ ]:




