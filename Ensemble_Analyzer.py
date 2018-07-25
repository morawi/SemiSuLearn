#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:17:37 2018

@author: malrawi
"""
import numpy as np
import os
from my_data import get_the_data
from some_functions import get_prm_from_file, get_labels_from_file, get_post_probs_from_file, get_accuracy_from_file, ensemble_prediction
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan) # to show all print out of an array


'''  # image display
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
plt.figure(dpi=700)
plt.plot(...)
idx= 1123; print("Label is", test_set.test_labels[idx]);plt.imshow(test_set.test_data[idx]);

to search for vgg1 in 
cnn_type =file_names_pred[1].find("vgg11")
this should return -1, if vgg11 is not in the file name, or a positive integer indicating the strat of vgg11


'''
data_set_list = [ 'STL10' 'EMNIST' , 'Cifar100', 'Cifar10', 'SVHN']
data_set_name = 'EMNIST' 
cwd = os.getcwd()
file_name =  'STL10_trn4trn_temp_pret_vgg19bn_new'#  '/Results_STL10/STL10_trn4trn_temp_pert' #  ./Results_EMNIST/results_vgg19_BN_temp_ensemble' #  './results' # "Dropout_tests/C100_Dropout_pt5" # _tst_then_train142 classifiers"
folder_name = './all_results/Results_EMNIST/Resnet151_tst_for_trn' # 'results'; './all_results/Results_' + data_set_name +'/' + file_name  
disp_images = True 
disp_correct  = False
test_for_train = True
post_prob_analysis = False
ensemble_average_softmax_prob = False
no_images_to_display =  400


def display_images_with_high_votes(max_val_unique, labels_voting, idx_all):
        
#    yy = np.array([], int)
#    for m_val in max_val_unique:
#        for xx in idx_all:
#            zz= labels_voting[xx]
#            if  np.max(zz)==m_val:  # any(zz==m_val):                      
#                yy = np.append(yy, xx)               
    
    zz= np.array([], int)
    for m_val in max_val_unique:
        match = np.argwhere(labels_voting == m_val)
        res = match[np.in1d(match[:, 0], idx_all)]
        zz= np.append(zz, res[:,0])
        
    cnt=1           
    for i in range(len(zz)):        
        idx = zz[i]
        if cnt > no_images_to_display: return 
        print("----------------------------------- ")
                       
        plt.figure(cnt); cnt=cnt+1                
        plt.xticks([])
        plt.yticks([])
        if test_for_train:
            if data_set_name=="SVHN" or data_set_name=="STL10":                    
                plt.imshow(np.rot90(np.fliplr(train_set.data[idx].T)))
                print(data_set_name, " Label is:", train_set.labels[idx].item(), "; which is", classes[train_set.labels[idx]])                                           
            elif  data_set_name=="EMNIST": 
                plt.imshow(np.rot90(np.fliplr(train_set.train_data[idx])))                                        
                print(data_set_name, " Label is:", train_set.train_labels[idx].item(), "; which is", classes[train_set.train_labels[idx]])                 
            else:
                plt.imshow(train_set.train_data[idx]);                                       
            
        else: 
            if data_set_name == 'SVHN' or data_set_name=="STL10":                        
                plt.imshow(np.rot90(np.fliplr(test_set.data[idx].T)))
                print(data_set_name, " Label is:", test_set.labels[idx].item(), "; which is", classes[test_set.labels[idx]]) 
            elif  data_set_name=="EMNIST": 
                plt.imshow(np.rot90(np.fliplr(test_set.test_data[idx])))
                print(data_set_name, " Label is:", test_set.test_labels[idx].item(), "; which is", classes[test_set.test_labels[idx]])
            else: 
                plt.imshow(test_set.test_data[idx]);                      
                            
        print("Majority_voting_predicted_label:", labels_pred[idx], "; which is", classes[labels_pred[idx]])                
        print("Index in ", data_set_name, " is: ", idx)
        print("Probability is: ",  m_val/no_of_classifiers)
        plt.show()
        print("----------------------------------- ")
           
if data_set_name=='STL10':
    train_set, test_set, classes, input_size, unlabeled_set = get_the_data(data_set_name) 
    test_set = unlabeled_set
    del unlabeled_set
        
else: 
    train_set, test_set, classes, input_size = get_the_data(data_set_name)

if test_for_train:    
    if data_set_name=='EMNIST':
        labels_target = train_set.train_labels
    else:                    
        labels_target = train_set.labels
else:
    if data_set_name=='EMNIST':
        labels_target = test_set.test_labels    
    else:
        labels_target = test_set.labels    
  
os.chdir(folder_name)
if data_set_name=='EMNIST':
    no_of_classes = max(train_set.train_labels)+1
else:
    no_of_classes = max(train_set.labels)+1 
loss, execution_time, no_train_accuracy =  get_prm_from_file()
labels, labels_voting = get_labels_from_file(no_of_classes)
acc, acc_of_all = get_accuracy_from_file()
no_of_images = np.size(labels, 1)
no_of_classifiers, no_of_images = labels.shape

if data_set_name=='EMNIST':
    labels_pred, max_val  = ensemble_prediction(labels_voting )
else:
    incorrect = (labels_target != labels_pred) 
incorrect = (labels_target.numpy() != labels_pred)
max_val_incorrect_counts = max_val[incorrect]
max_val_correct_counts = max_val[~(incorrect)]
idx_incorrect = np.arange(0, no_of_images, dtype=np.int); 
idx_correct = np.arange(0, no_of_images, dtype=np.int)
idx_incorrect = idx_incorrect[incorrect] # index's of incorrect samples
idx_correct = idx_correct[~incorrect]

plt.rcParams.update({'font.size': 16})
if post_prob_analysis:
    post_prob = get_post_probs_from_file(no_of_classes)
    if ensemble_average_softmax_prob:
        post_prob = np.mean(post_prob, axis=0)  # another way to take the mean of Softmax for the ensemble
    else: 
        x=1
        # post_prob = post_prob[-1,:] # getting only the latest classifier Softmax probablities      
         
    plt.figure(1); plt.hist( post_prob[incorrect], 200)  
    plt.figure(2); plt.hist( post_prob[~(incorrect)], 200)
    p_inco = sum(post_prob[incorrect] ==1)/no_of_images
    p_co = sum(post_prob[~incorrect] ==1 )/no_of_images
    
else:
    
    p_inco =sum(max_val_incorrect_counts== no_of_classifiers)/no_of_images
    p_co = sum(max_val== no_of_classifiers)/no_of_images
    plt.figure(3); plt.hist( max_val_incorrect_counts, 200)  ## n, bins, patches = plt.hist(s, 400, density=True)
    plt.figure(4); plt.hist( max_val_correct_counts, 200)
    print('No of classifiers used in the jury=', no_of_classifiers, '; acc of all the jury= ', sum((~incorrect))/no_of_images )
    print("Average classification accuracy", np.mean(acc))
    print("STd of classification accuracy", np.std(acc)) 
    
print('Incorrect samples that got recognized by all the CNNs', no_of_images*p_inco)
print("Correct samples that got recognized by all the CNNs", no_of_images*p_co)
print("OR is:",  p_inco*(1- p_co)/(p_co*(1-p_inco))  )
    

#plt.figure(5); plt.plot(acc)
#plt.figure(6); plt.plot(loss)
#plt.figure(7); plt.plot(execution_time)
#plt.figure(8); plt.plot(no_train_accuracy)
#plt.figure(9); plt.plot(acc_of_all.transpose())

plt.show()
if disp_images:    
    if post_prob_analysis:
        post_prob_unique = -np.unique(np.sort(-post_prob))  
        display_images_with_high_votes(post_prob_unique[0:1], post_prob, idx_incorrect)
        # we need to find which idx_incorrect has post_prob=1
        # we need a different fucntion for Softmax
    else: 
        if disp_correct: # Display Correct ones        
            max_val_correct_counts_unique = -np.unique(np.sort(-max_val_correct_counts))
            display_images_with_high_votes(max_val_correct_counts_unique[0:1], labels_voting, idx_correct)
        else:  # display InCorrect ones    
            max_val_incorrect_counts_unique = -np.unique(np.sort(-max_val_incorrect_counts))
            display_images_with_high_votes(max_val_incorrect_counts_unique[0:1] , labels_voting, idx_incorrect)
    
os.chdir(cwd)












