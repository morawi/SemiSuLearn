#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:39:00 2018

@author: malrawi
"""

"""


"""

# more info at:    http://pytorch.org/docs/master/torchvision/datasets.html
# https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py

import torchvision
import torchvision.transforms as transforms
import os
import json
import numpy as np

folder_of_data = '/home/malrawi/Desktop/My Programs/all_data/data'

def get_the_transform():
    mean = (0.49139968, 0.48215841, 0.44653091) # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
    std = (0.24703223, 0.24348513, 0.26158784)              
           
    """
    I think we need to consider removing RandomCrop and RandomHorizontalFlip
    """
    transform = transforms.Compose([        
        transforms.RandomCrop(32, padding=4), # added
        transforms.RandomHorizontalFlip(),        # added
        transforms.ToTensor(),       
        transforms.Normalize(mean, std),
    ])
    
    return transform


def get_the_data(data_set_name): 
    transform = get_the_transform()
    the_root = folder_of_data + data_set_name
    #Cifar10
    if data_set_name=='Cifar10':      
       
        train_set = torchvision.datasets.CIFAR10(
            root=the_root, train=True, download=True, transform=transform
        )
        
        test_set = torchvision.datasets.CIFAR10(
            root=the_root, train=False, download=True, transform=transform
        )
        
        classes = (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
        )
        
        
    # Cifar100    
    elif data_set_name=='Cifar100':
        
        train_set = torchvision.datasets.CIFAR100(
                root = the_root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(
                root=the_root, train=False, download=True, transform=transform)
        
        classes= (
           'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
            )
     
        
        
        "----------------------------------------------------------------------------------------------------"         
    #Street View House Number
    elif data_set_name=='SVHN':      
        
        train_set = torchvision.datasets.SVHN(
                root=the_root, split='train', download=True, transform=transform, target_transform=None
        )        
        
        test_set = torchvision.datasets.SVHN(
                root=the_root, split='test', download=True, transform=transform, target_transform=None)
        
        extra_set = torchvision.datasets.SVHN(
                root=the_root, split='extra', download=True, transform=transform, target_transform=None)
        
        classes = ('0','1','2', '3', '4', '5', '6', '7', '8', '9')
                 
        
        
        "----------------------------------------------------------------------------------------------------"                 
    # The STL-10 dataset image recognition datase
    # http://ai.stanford.edu/~acoates/stl10/
    elif data_set_name == 'STL10':    
        mean = (0.5, 0.5, 0.5) # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
        std = (0.25 , 0.25, 0.25 )                   # 0.1307, 0.3081,       
        transform = transforms.Compose([         
                transforms.ToTensor(),                    
                transforms.Normalize(mean, std),
                 ])
        
        train_set = torchvision.datasets.STL10(
                root=the_root, split='train', download=True, transform=transform, target_transform=None                
        )        
                
        test_set = torchvision.datasets.STL10(
                root=the_root, split='test', download=True, transform=transform, target_transform=None
        )
        
        # other split flags: ‘unlabeled’, ‘train+unlabeled’
        unlabeled_set = torchvision.datasets.STL10(
                root=the_root, split= 'unlabeled', download=True, transform=transform, target_transform=None
        )
        classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck',)
                    # 'OL1', 'OL2', 'OL3', 'OL4', 'OL5', 'OL6')
        labels = np.random.randint(0, len(classes), size=len(unlabeled_set.labels), dtype='int') # assigning randoms labels
        # labels =  np.loadtxt(the_root+'/unlabeled_labels.prd')
        unlabeled_set.labels = labels.astype(int) # dtype=np.uint8
        
         
   
    #    EMNIST ByClass:		814,255 characters. 62 unbalanced classes.
    #    EMNIST ByMerge: 	814,255 characters. 47 unbalanced classes.
    #    EMNIST Balanced:	131,600 characters. 47 balanced classes.
    #    EMNIST Letters:		145,600 characters. 26 balanced classes.
    #    EMNIST Digits:		280,000 characters. 10 balanced classes.
    #    EMNIST MNIST:		 70,000 characters. 10 balanced classes.
    # https://www.nist.gov/itl/iad/image-group/emnist-dataset    
    elif data_set_name == 'EMNIST':
        split_method= 'byclass'
        mean = (0.5, 0.5, 0.5) # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
        std = (0.25 , 0.25, 0.25 )                   # 0.1307, 0.3081,
        emnist_size = (32,32)
        transform = transforms.Compose([          
                transforms.Resize(emnist_size),
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),  
                # transforms.Lambda(lambda x:  torch.stack([x,x,x],2)),                  
               # transforms.Lambda(lambda x: x.permute(1,2,0)),             
#                transforms.Normalize(mean, std),
                 ])
        train_set = torchvision.datasets.EMNIST(
                root=the_root, train= True, split=split_method, download=True, 
                transform=transform, target_transform=None
        )        
                
        test_set = torchvision.datasets.EMNIST(
                root=the_root, train=False, split=split_method, download=True, 
                transform=transform, target_transform=None
        )
        # classes: 'byclass'
        if split_method=='byclass':
            numerals  = tuple( map(str, range(9 + 1)))
            alphabet_lower = tuple( map(chr, range(97, 123)))
            alphabet_upper = tuple(map(chr, range(65, 91)))
            classes = numerals + alphabet_upper + alphabet_lower
        elif split_method== 'digits' or split_method== 'mnist':
            classes = tuple( map(str, range(9 + 1)))
        elif split_method=='letters':            
            classes = tuple( map(chr, range(97, 123)))
        else: 
            classes = classes = tuple( map(str, range(47 + 1)))
        
    #  ImageNet
    # Download ImageNet64 from: http://image-net.org/download-images, store into train and val folders         
            
    elif data_set_name=='ImageNet64':
        root_dir=the_root
        traindir = os.path.join(root_dir, 'train')
        valdir = os.path.join(root_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
        train_set = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]))
                           
        test_set = torchvision.datasets.ImageFolder(
                valdir, 
                transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize]))
     
    # downlaod imagenet classes' names from 
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json       
        file_name='./all_data/data_imgnet64/imagenet_class_index.json'
        with open(file_name) as f:
            classes = json.load(f)
        f.close()
        classes = tuple(classes.values()) # Dictionary to Tuple
    else:
        print("Choose a dataset name, one of: Cifar10, Cifar100, SVHN, ImageNet64")            
   
    
    ''' NOW,  Returning the Data '''
    if data_set_name == 'EMNIST':
        input_size = emnist_size
    elif data_set_name == 'STL10' or data_set_name == 'SVHN':
        input_size = train_set.data[1].shape[1:3]        
    else: 
        input_size = train_set.train_data[1].shape[0:2]
    
    if data_set_name == 'STL10':
        return train_set, test_set, unlabeled_set, classes, input_size
    elif data_set_name=='SVHN':
        return train_set, test_set, extra_set, classes, input_size
    else:
        return train_set, test_set, classes, input_size
 
    # a fourth argument should be used with STL10 if unlabeled data are needed   
    # return train_set, test_set, classes, input_size, unlabeld_set




"""
    elif data_set_name == 'PhotoTour':    
        # for names in PhotoTour: https://pytorch.org/docs/stable/_modules/torchvision/datasets/phototour.html
        train_set = torchvision.datasets.PhotoTour(
                root='./dataPhotoTour', train=True, download=True, name='Notre Dame', transform=transform )        
        
        test_set = torchvision.datasets.PhotoTour(
                root='./dataPhotoTour', test=True, download=True, transform=transform )
        
        classes = ()
        
 """
 