# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:23:38 2019

@author: jsoto
"""

# Load libraries
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import graphviz


##------------------------------ 
#%%
#Function deffinition

#Function to create a baseline
def baseline_als(y, lam, p, niter):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

#Function to transform the wavenumber in raman 532 to wavelength
def length (wavenumber):
    wavelength= 1.0/(1.0/532-wavenumber/10**7)
    return wavelength

#Function so select the vector position of the wavenumber in raman 532
def ramvectpos(wavenumber):
    data_in = 32.6817
    data_fi = 3777.812
    vec_length = 1599
    vectpos = (length(wavenumber) - length(data_in))*vec_length/(length(data_fi)-length(data_in)) 
    vectpos = int(vectpos)
    return vectpos
    
    
#%%
#data read  
#First, we create a list with all the samples we have
filelist = glob.glob('*.txt')

#glob does not order it so we do it with .sort()
filelist.sort()

#First we select de range we want to use. We want to use several ranges:
start=ramvectpos(700)
end = ramvectpos(1800)
data_range =end - start

#Now we create a matrix with the data
data = np.zeros((len(filelist),data_range))
dataprime = np.zeros((len(filelist),data_range-1))

# And we create a matrix with the labels
label=np.empty([len(filelist), 4], dtype='U25')
counter = 0

#We read each file
for patient in filelist: 
    a = np.loadtxt(patient)
    s = a[start:end,1]
    L = a[start:end,0]
      
    
    
    #PREPROCESING
    #smoothing
    s=savgol_filter(s,31,3)
     
    #baseline substraction  
    b=baseline_als(s,1000000,0.001,40)
    s=s-b
   
        
    #normalization
    s=(s-np.mean(s))/np.std(s)
   
    #We add each sample to the matrix and add the labels to the other matrix
    data[counter,:] = s[:]
    label[counter,0] = patient[0:3] #Day
    label[counter,1]= patient[4:6] #Tecnique
    label[counter,2]= patient[7:12] #Patient
    if patient[16]=="A":
        label[counter,3]= "Prodromal state"
    elif patient[16] == "N":
        label[counter,3]= "Normal" 
    elif patient[16] == "O":
        label[counter,3]= "Other disease"
    elif patient[16] == "P":
        label[counter,3]= "Preclinical Alzheimer"
    
#The matrix with the data has its values in each row

# First derivative
    sprime = np.diff(s)/np.diff(L)
    dataprime[counter,:] = sprime[:]    
    
    counter += 1
#%%
##------class definition---------------
#discrimination state of the patient 
C=label[:,3]
#discrimination between days
#C=label[:,0]
#discrimination between patients
#C=label[:,2]

u, clases= np.unique(C, return_inverse=True)
n_clases=len(u)

#%%
'''
X gives me all the rows that are true for class == i. Then, I plot every line.
As I do not now how to creat one label for every class what I do is to 
average all of them and create a label for the average
'''
colors = cm.rainbow(np.linspace(0,1,n_clases))
fig = plt.figure(figsize=(12,7))
mainarea = fig.add_subplot(111)
mainarea.set_ylabel('Raman intensity')
mainarea.set_xlabel('Wavenumber (cm-1)')
for i in range(n_clases):
    X = data[clases == i,:]
    for j in range(len(X)):
        mainarea.plot(L,X[j,:],'--',color=colors[i],linewidth = 0.4)
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],linewidth=1, label=u[i])     
mainarea.legend()
plt.show()       
#fig.savefig('region1550-1750.png') 


#Here I just plot the average
colors = cm.rainbow(np.linspace(0,1,n_clases))
fig = plt.figure(figsize=(12,7))
mainarea = fig.add_subplot(111)
mainarea.set_ylabel('Raman intensity')
mainarea.set_xlabel('Wavenumber (cm-1)')
for i in range(n_clases):
    X = data[clases == i,:]
    xprom = np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],linewidth=0.4, label=u[i])
mainarea.legend()
plt.show()
#fig.savefig('average1550-1750.png')


#%%
fig = plt.figure(figsize=(12,7))
fig.suptitle('Different regions of interest')

ax1 = fig.add_subplot(221)
ax1.set_xlim(1720,1620)
ax1.set_xlabel('Wavenumber (cm-1)')
plt.setp(ax1.get_yticklabels(), visible=False)
ax1.tick_params(axis='both', which='both', length=0)
for i in np.arange(0,n_clases):
    X=data[clases==i,:]
    for j in np.arange(0,len(X)):
        ax1.plot(L,X[j,:],'--',color=colors[i],linewidth=0.4)
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],label=u[i],linewidth=0.4)
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.set_xlim(1000,900)
ax2.set_xlabel('Wavenumber (cm-1)')
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.tick_params(axis='both', which='both', length=0)
for i in np.arange(0,n_clases):
    X=data[clases==i,:]
    for j in np.arange(0,len(X)):
        ax2.plot(L,X[j,:],'--',color=colors[i],linewidth=0.4)
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],label=u[i],linewidth=0.4)
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.set_xlim(790,720)
ax3.set_xlabel('Wavenumber (cm-1)')
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.tick_params(axis='both', which='both', length=0)
for i in np.arange(0,n_clases):
    X=data[clases==i,:]
    for j in np.arange(0,len(X)):
        ax3.plot(L,X[j,:],'--',color=colors[i],linewidth=0.4)
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],label=u[i],linewidth=0.4)
ax3.legend()

ax4 = fig.add_subplot(224)
ax4.set_xlim(450,380)
ax4.set_xlabel('Wavenumber (cm-1)')
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.tick_params(axis='both', which='both', length=0)
for i in np.arange(0,n_clases):
    X=data[clases==i,:]
    for j in np.arange(0,len(X)):
        ax4.plot(L,X[j,:],'--',color=colors[i],linewidth=0.4)
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],label=u[i],linewidth=0.4)
ax4.legend()
plt.show()

#%% 
#The same but just with the average

fig = plt.figure(figsize=(12,7))
fig.suptitle('Different regions of interest')

ax1 = fig.add_subplot(221)
ax1.set_xlim(1720,1620)
ax1.set_xlabel('Wavenumber (cm-1)')
plt.setp(ax1.get_yticklabels(), visible=False)
ax1.tick_params(axis='both', which='both', length=0)
for i in np.arange(0,n_clases):
    X=data[clases==i,:]
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],label=u[i],linewidth=1.1)
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.set_xlim(1000,900)
ax2.set_xlabel('Wavenumber (cm-1)')
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.tick_params(axis='both', which='both', length=0)
for i in np.arange(0,n_clases):
    X=data[clases==i,:]
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],label=u[i],linewidth=1.1)
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.set_xlim(790,720)
#ax3.set_ylim(-0.8,0.4)
ax3.set_xlabel('Wavenumber (cm-1)')
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.tick_params(axis='both', which='both', length=0)
for i in np.arange(0,n_clases):
    X=data[clases==i,:]
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],label=u[i],linewidth=1.1)
ax3.legend()

ax4 = fig.add_subplot(224)
ax4.set_xlim(450,380)
ax4.set_xlabel('Wavenumber (cm-1)')
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.tick_params(axis='both', which='both', length=0)
for i in np.arange(0,n_clases):
    X=data[clases==i,:]
    xprom=np.mean(X,0)
    plt.plot(L,xprom,color=colors[i],label=u[i],linewidth=1.1)
ax4.legend()
plt.show()


#%%
#------------------------------------------------------------------------------
#------------------------------PCA---------------------------
n_samples,n_features=X.shape
#ncomponents = min(n_samples, n_features)
ncomponents = 10
pca = PCA(n_components=ncomponents)
X = pca.fit_transform(data)
print(pca.explained_variance_ratio_)

#%%
#Plotting the PCA

colores = cm.rainbow(np.linspace(0, 1, n_clases))
fig=plt.figure(figsize=(12,10))

#We plot PC1 vs PC2
ax1 = fig.add_subplot(221)
for color, i, target_name in zip(colores, np.arange(0,n_clases,1), u):
    xs=X[clases == i, 0]
    ys=X[clases == i, 1]

    plt.scatter(xs,ys,color=color,s=8,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PC2 vs PC1')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')


 
#We plot PC1 vs PC3
ax2 = fig.add_subplot(222)
for color, i, target_name in zip(colores, np.arange(0,n_clases,1), u):
    xs=X[clases == i, 0]
    ys=X[clases == i, 2]
    plt.scatter(xs,ys,color=color,s=8,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PC3 vs PC1')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC3')
 
#We plot PC2 vs PC3
ax3 = fig.add_subplot(223)
for color, i, target_name in zip(colores, np.arange(0,n_clases,1), u):
    xs=X[clases == i, 1]
    ys=X[clases == i, 2]
    plt.scatter(xs,ys,color=color,s=8,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PC3 vs PC2')
ax3.set_xlabel('PC2')
ax3.set_ylabel('PC3') 

plt.show()
#fig.savefig('PCA1550-1750.png')
#%%
#------------------------------------------------------------------------------
#------------------------------LDA---------------------------
n_samples,n_features=X.shape
#ncomponents = min(n_samples, n_features)
ncomponents = 2
lda = LinearDiscriminantAnalysis(n_components=ncomponents)
y = C
X_lda = lda.fit_transform(X,y)
print(lda.explained_variance_ratio_)
lda.explained_variance_ratio_

#%%
#Plotting the LDA

colores = cm.rainbow(np.linspace(0, 1, n_clases))
fig=plt.figure(figsize=(12,10))

#We plot PC1 vs PC2
ax1 = fig.add_subplot(221)
for color, i, target_name in zip(colores, np.arange(0,n_clases,1), u):
    xs=X_lda[clases == i, 0]
    ys=X_lda[clases == i, 1]

    plt.scatter(xs,ys,color=color,s=8,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA2 vs LDA1')
ax1.set_xlabel('LDA1')
ax1.set_ylabel('LDA2')
#fig.savefig('LDApatients.png')
#%%
#Classification LDA
# =============================================================================
classification = DecisionTreeClassifier()
cross_val_score(classification,X_lda,y,cv=10)
#%%
#Prediction LDA
# =============================================================================
# Next, let’s see whether we can create a model to classify the using the LDA 
# components as features. First, we split the data into training and testing sets.
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, random_state=1)
# =============================================================================
# Then, we build and train a Decision Tree. After predicting the category of 
# each sample in the test set, we create a confusion matrix to evaluate the
# model’s performance.
# =============================================================================
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
cross_val_score(dt, X_lda, y, cv=10)


 
#tree.plot_tree(dt.fit(X_train, y_train))