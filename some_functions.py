#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:35:01 2018

@author: malrawi
"""
import numpy as np
import glob , os
from math import floor

def get_prm_from_file():
    file_names_prms = sorted( glob.glob("*.prm"), key=os.path.getatime)  # loading prediction files
    
    loss =  np.arange(0, len(file_names_prms), dtype=np.float)
    execution_time =  np.arange(0, len(file_names_prms), dtype=np.float)
    no_train_accuracy =  np.arange(0, len(file_names_prms), dtype=np.float)
    i=0
    for file_name in file_names_prms:            
        file_ = open(file_name, 'r')         
        xx = file_.readline(); 
        xx = np.fromstring(xx, dtype=float, sep='('); loss[i]=xx[0];         
        xx = file_.readline() # trivial_1
        xx = file_.readline(); 
        no_train_accuracy[i]= np.float64(xx[22:-1])
        xx = file_.readline() # trivial_2
        xx= file_.readline(); 
        execution_time[i] =  np.float64(xx[16:-1])             
        file_.close()
        i=i+1
        
    return loss, execution_time, no_train_accuracy 


def get_labels_from_file(no_of_classes, del_after_use=False):
    labels = np.array([])    
    file_names_pred = sorted(glob.glob('*.prd'), key=os.path.getmtime)
    for file in file_names_pred:             
        labels = np.append(labels, np.loadtxt(file), axis=0 ) 
        if (del_after_use==True):
            os.remove(file)
    
    labels, labels_voting = count_votes(no_of_classes, labels, len(file_names_pred))
    
#    labels = labels.reshape([len(file_names_pred), floor(len(labels)/len(file_names_pred)) ] )    
#    labels_voting = np.zeros( shape=( np.size(labels, 1), no_of_classes ) )
#    for i in range(0, no_of_classes):
#        tmp = np.sum(labels==i, 0)
#        labels_voting[:,i]=tmp
    
    return  labels, labels_voting

def count_votes(no_of_classes, labels, no_labels_vectors):
    labels = labels.reshape([no_labels_vectors, floor(len(labels)/ no_labels_vectors) ] )
    labels_voting = np.zeros( shape=( np.size(labels, 1), no_of_classes ) )
    for i in range(0, no_of_classes):
        tmp = np.sum(labels==i, 0)
        labels_voting[:,i]=tmp
    return labels, labels_voting



def get_post_probs_from_file(no_of_classes):
    probs = np.array([])
    file_names_prb = sorted( glob.glob("*.prb"), key=os.path.getatime)  # loading prediction files
    for file in file_names_prb:    
        probs = np.append(probs, np.loadtxt(file), axis=0 ) 
    probs =np.squeeze( probs.reshape([len(file_names_prb), floor(len(probs)/len(file_names_prb)) ] ))
        
#    all_post_probs = np.zeros( shape=( np.size(probs, 1), no_of_classes ) )
#    for i in range(0, no_of_classes):
#        tmp = np.sum(labels==i, 0)
#        labels_of_all_classes[:,i]=tmp
    return  probs

def get_accuracy_from_file():    
    file_names_acc= sorted(glob.glob('*.acc'), key=os.path.getmtime)
    acc_all = np.array([])
    for file in file_names_acc:
    #    print(file)    
        acc_all = np.append(acc_all, np.loadtxt(file), axis=0 ) 
    acc_all = acc_all.reshape([len(file_names_acc), floor(len(acc_all)/len(file_names_acc)) ] )
    acc0 = acc_all[:,-1]
        
    return  acc0, acc_all

# no_of_images is total number of image examples/samples in the dataset
def ensemble_prediction(labels_voting ):  
    no_of_samples = np.size(labels_voting,0)
    labels_pred = np.zeros(no_of_samples, int )
    max_val = np.zeros( no_of_samples, int )
    for i in range(0, no_of_samples ):
        labels_pred[i] = np.argmax(labels_voting[i,:])
        max_val[i] = np.max(labels_voting[i,:])
    
    labels_pred = labels_pred.astype(int)  # precaution
    return   labels_pred, max_val  



def bincount2D_vectorized(a):    
    N = a.max()+1
    a_offs = a + np.arange(a.shape[0])[:,None]*N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1,N)

# one sample t-test
# https://plot.ly/python/t-test/
# true_mu = 0
# onesample_results = scipy.stats.ttest_1samp(data1, true_mu)


