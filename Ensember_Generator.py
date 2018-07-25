#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:31:13 2018

@author: malrawi
"""

""" DCNN on Different Datasets

Based on:
- https://github.com/pytorch/
- http://pytorch.org/docs/master/optim.html
- https://github.com/creafz/pytorch-cnn-finetune/
- https://github.com/kuangliu/pytorch-cifar
-  (DLoader) https://www.youtube.com/watch?v=zN49HdDxHi8
 - https://discuss.pytorch.org/t/data-augmentation-in-pytorch/7925
 https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
 https://github.com/Cadene/pretrained-models.pytorch/blob/master/README.md
 https://github.com/probprog/CSCS-summer-school-2017/blob/master/exercises/exercise-2-pytorch/CSCS-summer-school-2017-exercise-2.ipynb
 http://www.scholarpedia.org/article/Ensemble_learning
 https://discuss.pytorch.org/t/how-to-implement-oversampling-in-cifar-10/16964/15  # get a sample from the data
 https://towardsdatascience.com/the-softmax-function-neural-net-outputs-as-probabilities-and-ensemble-classifiers-9bd94d75932
 http://cs231n.github.io/convolutional-networks/

https://github.com/pytorch/examples/tree/master/fast_neural_style/images  # Neural art and styles
https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b
https://towardsdatascience.com/stochastic-weight-averaging-a-new-way-to-get-state-of-the-art-results-in-deep-learning-c639ccf36a
https://www.slideshare.net/gabrielspmoreira/feature-engineering-getting-most-out-of-data-for-predictive-models
https://colab.research.google.com/notebooks/basic_features_overview.ipynb#scrollTo=Wej_mEyXQSHc
https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
https://www.programcreek.com/python/example/89637/torch.utils.data.sampler.SubsetRandomSampler
https://www.datascience.com/blog/production-level-code-for-data-science
https://kivy.org/#home
https://www.youtube.com/watch?v=hM9qbW8-roE
"""


"""
##### From [torchvision](https://github.com/pytorch/vision/) package:

- ResNet (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`)
- DenseNet (`densenet121`, `densenet169`, `densenet201`, `densenet161`)
- Inception v3 (`inception_v3`)
- VGG (`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`)
- SqueezeNet (`squeezenet1_0`, `squeezenet1_1`)
- AlexNet (`alexnet`)

##### From [Pretrained models for PyTorch](https://github.com/Cadene/pretrained-models.pytorch) package:
- Dual Path Networks (`dpn68`, `dpn68b`, `dpn92`, `dpn98`, `dpn131`, `dpn107`)
- ResNeXt (`resnext101_32x4d`, `resnext101_64x4d`)
- NASNet (`nasnetalarge`)  ('nasnetamobile')
- Inception-ResNet v2 (`inceptionresnetv2')
- Inception v4 (`inception_v4`)
- Xception (`xception`)
"""

import gc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from cnn_finetune import make_model
import time
import numpy as np
from math import floor
import datetime
from my_data import get_the_data
import torch.nn.functional as F


# gpu_device = 1; torch.cuda.set_device(gpu_device)  # torch.cuda.set_device(device_num)
data_set_list = [ 'STL10' 'EMNIST' , 'Cifar100', 'Cifar10', 'SVHN']
data_set_name = 'STL10' #'EMNIST'  # 'Cifar10', Cifar100, 'ImageNet64'
pre_trained_model = True
use_test_then_train = False
train_original_classifier  = False
dropout_p = 0.4
lr = 0.01 
weight_decay = 5e-4
no_epochs = 120
momentum_val= 0.9
use_nestrov_moment = False 
change_batch_size = False
if change_batch_size: batch_size_start = 200 
#number_of_mini_bathces = 400 # For the current implementation, trn_batch_sz and test_batch_sz are multiplied  by 100 (magic_number) to attain the real numbers
# test_dataset_size = 10000
# train_dataset_size = 50000
trn_batch_sz = 250 # int( train_dataset_size/number_of_mini_bathces)  # we need to add if statement to change this to the size of the trn, depending on test_then_trn
tst_batch_sz =  125 # int( test_dataset_size/number_of_mini_bathces)
seed_value =  floor(time.time())   ###################### default is 1
my_model = 'dummy'


parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
parser.add_argument('--batch-size', type=int, default=trn_batch_sz, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default= tst_batch_sz, metavar='N',
                    help='input batch size for testing (default: 64)')

parser.add_argument('--use_test_then_train', type=bool, default=use_test_then_train, metavar='B',
                    help='input true or false (default: True)')

parser.add_argument('--epochs', type=int, default= no_epochs, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=lr, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=momentum_val, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA trainingalexnet')
parser.add_argument('--seed', type=int, default=seed_value, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help ='how many batches to wait before logging training status')

parser.add_argument('--dropout-p', type=float, default= dropout_p, metavar='D',
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--model-name', type=str, default= my_model, metavar='M',
                    help='model name (default: ' + my_model+')')

parser.add_argument('--train_original_classifier', type=bool, default=train_original_classifier, metavar='S',
                    help='input true or false (default: False)')

parser.add_argument('--change_batch_size', type=bool, default= change_batch_size, metavar='S',
                    help='input true or false (default: False)')

parser.add_argument('--weight-decay', type=float, default=weight_decay,
                    metavar='W', help='weight detorch.optim.cay (default: 1e-4)')

parser.add_argument('--pre-trained-model', type=bool, default= pre_trained_model, 
                    metavar='W', help='input true or false (default: False)')

parser.add_argument('--data-set-name', type=str, default= data_set_name, 
                    metavar='S', help='Name of dataset used')

args = parser.parse_args()


args.use_temporal_ensemble = True
args.temporal_ensemble_frequency_save = 1
args.temporal_perturbation_f = 3 # should be prime number
args.temporal_alpha = 6        
args.temporal_pert_skip = 8
args.lr_milestones= [20, 30, 80, 170, 290 ]
"""  temporal_alpha: this is used to construct the perturbation noise, random gaussian noiose added to the gradient,
higher values yield higher noise, temporal alpha will decay with epoch advance, thus, higher values 
may indicate more training time will be needed    
temporal_perturbation_f: The noise is added to the gradient every temporal_perturbation_f, should be prime number
temporal_ensemble_frequency_save: When to save the classifier, 1 means at every epoch
temporal_pert_skip: The number of epochs to skip before performing temporal perturbation, could be low if one
uses a pre-trained model (as the convergence could be fast)
    """




# args.cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

print('houdi')
print(args)

def train(epoch, train_loader):
    total_loss = 0
    total_size = 0
    model.train()    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if args.use_temporal_ensemble and epoch>args.temporal_pert_skip and not(epoch % args.temporal_perturbation_f):    
            print(".", end="")           
            output = output+ args.temporal_alpha* torch.rand(output.shape).cuda()/np.sqrt(epoch+1)
            
        loss = criterion(output, target)       
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))
    return total_loss / total_size



def test(test_loader):
    if not hasattr(test, "counter"):
        test.counter = 0  # it doesn't exist yet, so initialize it
    test.counter += 1
    model.eval()
    test_loss = 0
    correct = 0
    predicted_labels =[]; predicted_labels = np.array(predicted_labels, int)  # declare an empty array       
          
    
    
    output_posterior_prob =[]; output_posterior_prob = np.array(output_posterior_prob, float)  # declare an empty array       
        
    
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()        
            predicted_labs =  np.squeeze( (pred.data).cpu().numpy())        
            predicted_labels = np.concatenate((predicted_labels, predicted_labs), axis=0) # we need to do concatnation, sicne data is devided into batches
            output = F.softmax(output, dim=1) # applying softmax
            output_probs =  (output.data).cpu().numpy()
            output_probs = np.amax(output_probs, axis=1)
            output_posterior_prob =  np.concatenate((output_posterior_prob, output_probs), axis=0) # we need to do concatnation, sicne data is devided into batches
                
    test_loss /= len(test_loader.dataset)
    accuracy_ = 100. * correct / len(test_loader.dataset)
    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy_))
    if args.use_temporal_ensemble and (test.counter% args.temporal_ensemble_frequency_save==0):
        acc = np.zeros(args.epochs, float); acc[-1]=accuracy_ # just fill with trivial zeros, and set last element to acc
        print('Savign snapshot ensemble')
        write_to_file(acc, predicted_labels, -1, -1, output_posterior_prob, args) # loss and no_train_accuracy are marked with -1, as they are not needed and here
    return accuracy_, predicted_labels, output_posterior_prob


def write_to_file(testing_accuracy, predicted_labels, no_training_acc, loss, output_posterior_prob, args):
    exec_time = time.time() - start_timer
    print("execution time: ", exec_time)     
    # print(testing_accuracy[-1], args)
    if args.use_test_then_train:
        exx = './results/'+ args.model_name+'_'+'test_for_training__' + str(datetime.datetime.now())
    else: 
        exx = './results/'+ args.model_name+'_'+'train_for_training__' + str(datetime.datetime.now())
    np.savetxt( exx+'_accuracy_.acc', testing_accuracy, delimiter=',', fmt='%.2f'  )    
    np.savetxt( exx+'._predicted_labels.prd', predicted_labels, delimiter=',', fmt='%d'  )
    np.savetxt( exx+'_posterior_.prb', output_posterior_prob, delimiter=',', fmt='%.2f'  )    
    text_file = open(exx+ '_parameters.prm', "w")
    text_file.write(str(loss) + '(\n Max/Last_Epoch_loss) \n' )
    text_file.write('No training accuracy: ' + str(no_training_acc) +'\n' )
    text_file.write(str(args)+'\n')
    text_file.write('Execution time: '+ str(exec_time))
    text_file.close()


""" -----------------------------------------------------------------------
 -------------------------------------------------------------------------- 
 acc: testing accuracy using the test set
 acc0: testing accuracy, before training
 predicted labels: predicted labels of the last epoch
 
 """
def run_net_fixed_batch(total_epochs):  
       
    if args.use_test_then_train:    # Z Learning - hour accuracy, results without training the network
        train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=False, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, args.test_batch_size, shuffle=True, num_workers=2) 
        acc_0, predicted_labels_0, output_posterior_prob = test(train_loader)            
    else:
        train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, args.test_batch_size, shuffle=False, num_workers=2)                 
        acc_0, predicted_labels_0, output_posterior_prob = test(test_loader)
           
    acc= np.zeros(total_epochs+1, float)    
    for epoch in range(1, total_epochs + 1):
        acc[epoch], predicted_labels, loss, output_posterior_prob = do_it(epoch, train_loader, test_loader, acc)
        if epoch>5 and acc[epoch]<15:
            acc[0]=0
            return acc, acc, acc, acc, acc # just a trivial return, to use acc as a flag to initialize the round
    acc = np.delete(acc, 0)     # removinnvalid syntag the first element as it is not used, index starts from 1                
    return acc, predicted_labels, acc_0, loss, output_posterior_prob   # predicted labels of the last epoch


def run_net_variable_batch(total_epochs):
    acc= np.arange(0, total_epochs+1, dtype=np.inceptionresnetv2float)    
    for epoch in range(1, total_epochs + 1):        
        train_batch_size, test_batch_size = get_batch_size(epoch)
        train_loader = torch.utils.data.DataLoader(train_set, train_batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, test_batch_size, shuffle=False, num_workers=2)
        acc[epoch], predicted_labels, loss, output_posterior_prob = do_it(epoch, train_loader, test_loader, acc)
    acc = np.delete(acc, 0)    
    return acc, predicted_labels # predicted labels of the last epoch



def get_batch_size(epoch):
    
    if args.use_test_then_train:
        test_batch_size = batch_size_start + ( floor(test_dataset_size/number_of_mini_bathces) - batch_size_start )* (epoch-1)/(args.epochs) # liner increment
        test_batch_size = floor(test_batch_size)
        train_batch_size= args.batch_size
        print(" batch size:", test_batch_size*number_of_mini_bathces)                       
        
    else:           
        train_batch_size= batch_size_start + (floor(train_dataset_size/number_of_mini_bathces) - batch_size_start )*(epoch-1)/(args.epochs) # linear increment
        train_batch_size = floor(train_batch_size)
        test_batch_size = args.test_batch_size
        print(" batch size:", train_batch_size*number_of_mini_bathces)    
    
    return train_batch_size, test_batch_size

""" --------------------------  END of Run Net Routines  -------------------- """

def do_it(epoch, train_loader, test_loader, acc):
            
    if args.use_test_then_train:
        loss= train(epoch, test_loader)            
        acc, predicted_labels, output_posterior_prob = test(train_loader)            
    else:            
        loss = train(epoch, train_loader)
        acc, predicted_labels, output_posterior_prob = test(test_loader)
    
    # scheduler.step(loss) # to be used with ReduceLROnPlateau
    scheduler.step();     print("lr = ", scheduler.get_lr(), '\n') # to be used with MultiStepLR
    
    
    return acc, predicted_labels, loss, output_posterior_prob


#  ------------------------   __main__ function   -----------------------------

dnn_list=['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
        'resnext101_32x4d', 'resnext101_64x4d', 'nasnetalarge',   
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'xception',        
        'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107',
        'densenet121', 'densenet169', 'densenet201', 'densenet161',       
        'squeezenet1_0', 'squeezenet1_1', 
        'alexnet',
        'resnext101_32x4d' , 'resnext101_64x4d'
        'nasnetalarge' , 'nasnetamobile',
        'inceptionresnetv2', 'inception_v3', 'inception_v4' ]

#  dnn_list = ['alexnet', 'inception_v3'] # these are not working


dnn_list = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
dnn_list = ['vgg19_bn']

if data_set_name == 'STL10':
    train_set, test_set, classes, input_size, unlabeled_set =get_the_data(data_set_name) 
    test_set = unlabeled_set
    del unlabeled_set
else:
    train_set, test_set, classes, input_size = get_the_data(data_set_name) 

for model_moujou in dnn_list:
    print('-----------------------', model_moujou, '-----------------------')
    args.model_name = model_moujou          
    for i in range(0, 1):
        print("Round ", i, "-th");   print('model', model_moujou)       
        model = make_model(model_moujou, 
                           num_classes=len(classes), 
                           pretrained=args.pre_trained_model,  
                           dropout_p=args.dropout_p, 
                           input_size = input_size,
                            )
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=use_nestrov_moment,
                              # dampening = 0.005
                              )               
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma= .5)                              
        start_timer = time.time()        
        if args.change_batch_size:   
            testing_accuracy, predicted_labels, no_training_acc, output_posterior_prob = run_net_fixed_batch(args.epochs)   # predicted_labels of the last epoch are returned       
            testing_accuracy1, predicted_labels = run_net_variable_batch(args.epochs)   
              
            testing_accuracy = np.concatenate((testing_accuracy1, testing_accuracy), axis=0)
        else:
            testing_accuracy, predicted_labels, no_training_acc, loss, output_posterior_prob  = run_net_fixed_batch(args.epochs) #using test for training, and train for testig      
            if testing_accuracy[0]==0:                 
                print("Skipping this round trapped in chance-level perforamance")
                continue
        print('no training accuracy: ', no_training_acc)
        write_to_file(testing_accuracy, predicted_labels, no_training_acc, loss, output_posterior_prob, args)
        print('\n')
        gc.collect()





'''
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                               verbose= True, factor=0.2)
        
        '''