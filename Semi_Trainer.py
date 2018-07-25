#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:31:13 2018

@author: malrawi
"""

""" DCNNs - Semi Supervised Learning

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


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DT
from cnn_finetune import make_model
import time
import numpy as np
import sys; sys.path.append('/home/malrawi/Desktop/My Programs/DCNN-1') # folder where my_data.py is
from my_data import get_the_data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from some_functions import ensemble_prediction, count_votes

# gpu_device = 1; torch.cuda.set_device(gpu_device)  # torch.cuda.set_device(device_num)

data_set_name = 'STL10' #'EMNIST'  # 'Cifar10', Cifar100, 'ImageNet64'

no_epochs = 60
semi_sup_iterations = 4
epoch_skip = 15  # should be high if not using a pre-trained model
temporal_pert_skip = 5 # if this is set to low value, it will affect the convergence
pre_trained_model = False

lr = 0.01
weight_decay = 5e-4
dropout_p = 0.4
momentum_val= 0.9
use_nestrov_moment = True 
trn_batch_sz = 64 # int( train_dataset_size/number_of_mini_bathces)  # we need to add if statement to change this to the size of the trn, depending on test_then_trn
tst_batch_sz = 125 # int( test_dataset_size/number_of_mini_bathces)
seed_value = 1 # int(time.time())   ###################### default is 1
my_sampler_size = 1000


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
        'inceptionresnetv2', 'inception_v3', 'inception_v4']

my_model = 'vgg16_bn'
model_name_attention = ''# 'dpn92' # to remove this model, use model_name_attention=''


parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
parser.add_argument('--batch-size', type=int, default=trn_batch_sz, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default= tst_batch_sz, metavar='N',
                    help='input batch size for testing (default: 64)')

parser.add_argument('--epochs', type=int, default= no_epochs, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=lr, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=momentum_val, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA trainin')
parser.add_argument('--seed', type=int, default=seed_value, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                    help ='how many batches to wait before logging training status')

parser.add_argument('--dropout-p', type=float, default= dropout_p, metavar='D',
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--model-name', type=str, default= my_model, metavar='M',
                    help='model name (default: vgg1)')

parser.add_argument('--train_original_classifier', type=bool, default=False, metavar='S',
                    help='input true or false (default: False)')

parser.add_argument('--weight-decay', type=float, default=weight_decay,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--pre-trained-model', type=bool, default= pre_trained_model, 
                    metavar='W', help='input true or false (default: False)')


args = parser.parse_args()

args.epoch_skip = epoch_skip # number of epochs to skip before predicting unlabeled examples
args.semi_sup_iterations = semi_sup_iterations
args.temporal_pert_skip = temporal_pert_skip
args.lr_milestones= [50, 90, 250 ]  #[20, 40, 80 ]

args.use_temporal_ensemble = False 
args.temporal_perturbation_f = 3 # should be prime number
args.temporal_alpha = 0
args.model_name_attention = model_name_attention        

"""  temporal_alpha: this is used to construct the perturbation noise, random gaussian noiose added to the gradient,
higher values yield higher noise, temporal alpha will decay with epoch advance, thus, higher values 
may indicate more training time will be needed    
temporal_perturbation_f: The noise is added to the gradient every temporal_perturbation_f, should be prime number
temporal_pert_skip: The number of epochs to skip before performing temporal perturbation, could be low if one
uses a pre-trained model (as the convergence could be fast)
    """

def train(epoch, train_loader):
    print('Training, ', end="")
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
        loss.backward()
        optimizer.step()
        total_size += data.size(0)
        if batch_idx % args.log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.7f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),                
                100. * batch_idx* len(data) / len(train_loader.sampler), total_loss / total_size))
    return total_loss / total_size


def test(test_loader):  
    print('--- Test with:', test_loader.dataset.split, 'set ---', end="")
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
    
    test_loss /= len(test_loader.sampler)
    accuracy_ = 100. * correct / len(test_loader.sampler)    
    
    print(' Average loss is {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
      #  test_loss, correct, len(test_loader.dataset),
        test_loss, correct, len(test_loader.sampler),
        accuracy_))
    
    return accuracy_, predicted_labels, output_posterior_prob


def run_net(train_loader, test_loader, posteriori_method):   
    print('\n ---- Train with:', train_loader.dataset.split, 'set----')    

    acc= np.zeros(args.epochs, float); 
    loss= np.zeros(args.epochs, float)
    ensemble_labels = np.array([])
            
    for epoch in range(1, args.epochs):
        scheduler.step();     print("lr = ", scheduler.get_lr(), " ", end="") # to be used with MultiStepLR
        loss[epoch-1] = train(epoch, train_loader)            
        if posteriori_method!='softmax_prob':
            acc[epoch-1], predicted_labels, softmax_prob = test(test_loader)   
            # generating lables for ensemble building
            if epoch>args.epoch_skip:              
                ensemble_labels = np.append(ensemble_labels, predicted_labels, axis=0 )     
    if posteriori_method =='softmax_prob':
        acc[epoch-1], predicted_labels, softmax_prob = test(test_loader)
    # put everything together
    results = {'acc':acc, 'predicted_labels': predicted_labels, 
              'loss':loss, 'softmax_prob':softmax_prob,
              'ensemble_labels':ensemble_labels}
      
    # predicted labels of the last epoch
    return results 

""" --------------------------  END of Run Net Routines  -------------------- """

def build_the_ensemble(results):
    # build ensmble, if not using unlabeled data
    no_of_classes = len(unlabeled_loader.dataset.classes)
    no_of_ensembles = args.epochs-args.epoch_skip-1   # this is in fact the no of ensembles used             
    labels, labels_voting = count_votes(no_of_classes, results['ensemble_labels'], no_of_ensembles)
    print('diagnostics: no of ensembles...', no_of_ensembles)
    results['labels_voting']= labels_voting
    results ['no_of_ensembles'] = no_of_ensembles
    return results


def semi_supervised_train(unlabeled_set, unlabeled_loader):    
    # global unlabeled_set, unlabeled_loader    
     
    posteriori_method = 'ensemble_prob' #'ensemble_prob' #'softmax_prob' # choices: ensemble_prob or softmax_prob    
    # my_sampler_size = len(unlabeled_set)//4
    
    ''' if one choose soft_max, it is the last one returned by model(), same for the predicted labels
    However, any other value used will set the ensemble method, posteriori and labesl from the ensemble 
    '''
    
    for round_id in range(0, args.semi_sup_iterations):   
        
        print('\n ---------------------- Round ', round_id+1, '----------------------')
        print('...... Train model ', args.model_name, '......' )
        train_result = run_net( train_loader, unlabeled_loader, posteriori_method)
        if posteriori_method!='softmax_prob': train_result= build_the_ensemble(train_result)          
        pred_labels, posterior_prob = get_labels_and_prob(train_result, posteriori_method)                  
        my_sampler = get_the_sampler(posterior_prob, my_sampler_size, pred_labels, len(classes)) # my_sampler = get_the_sampler(train_result['softmax_prob'], p_threshold)      
        
        if round_id == 0: # at the beggening get all the labels, we dont the correct have my_sampler yet
            unlabeled_set.labels = pred_labels 
        else:         
            # otherwise, update the labels using my_sampler and posteriori
            unlabeled_set.labels[my_sampler.indices] =  pred_labels[my_sampler.indices] # assigning labels using relabel
         
        unlabeled_loader = DT.DataLoader(unlabeled_set, args.batch_size, sampler=my_sampler,
                                                    num_workers=2)                                 
                       
        unlabeled_result = run_net(unlabeled_loader, train_loader, posteriori_method) ## train with the unlabeled data
        
        #And now, we need to remove the sampler from the unlabeled_loader, so that we have all the data in the next stage
        unlabeled_loader = DT.DataLoader(unlabeled_set, args.batch_size, sampler=None, shuffle=False, 
                                                       num_workers=2) 
    # returning the loader with the last sampler
    unlabeled_loader = DT.DataLoader(unlabeled_set, args.batch_size, sampler=my_sampler,
                                                    num_workers=2)        
    return unlabeled_result, train_result, unlabeled_set, unlabeled_loader 


def get_labels_and_prob(results, posteriori_method):
    if posteriori_method == 'softmax_prob':
        labels = results['predicted_labels']
        posterior_prob = results['softmax_prob']
    else: # else, ensemble probs and labels         
        labels, max_val = ensemble_prediction(results['labels_voting'])                     
        posterior_prob = max_val/results['no_of_ensembles'] # or using softmax_posterior_prob               
    return labels, posterior_prob
        

# we need to use labels to keep the selector balanced    
def get_the_sampler(posterior_prob, sampler_size, labels, no_classes): 
    ''' 
    Args in - 
        score: posteriori values for each of the labels [0 to 1], 1 indicates 
        the label has high likliness to be of a correct class
        sampler_sie: the intended sampler size the user wants to get back, the 
        size of the returned sampler will be sligthly less than this
        labels: an array containing the labels
        no_classes: the number of classes in the problem        
        
    Parameters - 
        percentage_of_selected_samples: selecting 50% for the samples with the highest 
        'score' values 
    '''
    
    percentage_of_selected_samples = 50/100
    
    len_labels_per_class = np.zeros(no_classes, dtype=int)      
    idx_per_class = np.zeros([no_classes, len(labels)], dtype=int)
    for i in range(no_classes):
        idx_per_class[i] = labels==i
        len_labels_per_class[i] = sum(idx_per_class[i] == True)
    no_labels_per_class = min(len_labels_per_class)   
    sampler_pool_size = int( no_labels_per_class * percentage_of_selected_samples ) 
    
    sampler_size = int(sampler_size/no_classes)
    if(sampler_size >sampler_pool_size): 
        print('You need to decrease the value percentage_of_selected_samples: ', percentage_of_selected_samples)
        exit('Exiting function get_the_sampler(): sampler_size has become larger than sampler_pool_size')
        
    
    my_sampler = []
    for i in range(no_classes):
        sample_idx = (-posterior_prob[idx_per_class[i]]).argsort()[:sampler_pool_size]   
        sample_idx = np.random.permutation(sample_idx)
        sample_idx = sample_idx[:sampler_size]
        my_sampler.extend(sample_idx)
    
    if len(my_sampler) <100:  exit('Exiting function get_the_sampler(): small sampler')    
    my_sampler = torch.utils.data.sampler.SubsetRandomSampler(my_sampler)  
    return my_sampler


 
def train_dcnn_(epochs, train_loader): # build the classifier if not pre-trained one
    for epoch in range(1, epochs):
        train(epoch, train_loader)
   
    
def print_all_results():
     print_final_results(train_loader)
     print_final_results(test_loader)
     print_final_results(unlabeled_loader)

def print_final_results(the_loader):     
    acc1, predicted_labels1, output_posterior_prob1 = test(the_loader) 
    plt.title('loader-'+ train_loader.sampler.data_source.split +'-probability/confidence');
    # plt.plot(output_posterior_prob1)
    plt.hist( output_posterior_prob1, 200)  
    plt.show()
    
    
def construct_dcnn(model_name):
    global model, optimizer, scheduler, criterion
    model = make_model(model_name, num_classes=len(classes), 
                       pretrained=args.pre_trained_model,  
                       dropout_p=args.dropout_p, 
                       input_size = input_size)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                      weight_decay=args.weight_decay, nesterov=use_nestrov_moment)               # dampening = 0.005                      
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma= .95) 
    criterion = nn.CrossEntropyLoss()                                  

    
def get_the_loaders():
    
# # Selecting sub-set from the training, to see how much is enough     
#    no_of_samples = 3000
#    sample_idx = np.random.randint(0, 5000-1, no_of_samples, dtype='int') 
#    my_train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)
    
    # ................ Shuffle is set to false
    train_loader = DT.DataLoader(train_set, args.batch_size, shuffle=False, num_workers=2)
    
    test_loader = DT.DataLoader(test_set, args.test_batch_size, shuffle=False, num_workers=2)
    
    unlabeled_loader = DT.DataLoader(unlabeled_set, args.batch_size, sampler=None, shuffle=False, 
                                                   num_workers=2)
    return train_loader, test_loader, unlabeled_loader    
               


def add_new_rand_labels(input_set):
    input_set.labels_original = input_set.labels # keep the original labels
    input_set.labels = 1+ np.random.randint(0, 
                                    max(input_set.labels), size=len(input_set.labels), dtype='int') # assigning randoms labels
    return input_set
       

def random_seeding(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)    
    if use_cuda: torch.cuda.manual_seed_all(seed_value)
    


"""  ------------------------   -----------------   ----------------------------- """
"""  ------------------------   __main__ function   ----------------------------- """
"""  ------------------------   -----------------   ----------------------------- """

"Using random labels in the testing set, and train it, then putting back the original labels before testing"
start_timer = time.time()
print('##### ----- Semi-Supervised Learning using... ', data_set_name, '-----s#####')
print(args)
use_cuda = not(args.no_cuda) and torch.cuda.is_available() 
device = torch.device('cuda' if use_cuda else 'cpu')
random_seeding(seed_value, use_cuda)
train_set, test_set, unlabeled_set, classes, input_size = get_the_data(data_set_name) 
train_loader, test_loader, unlabeled_loader = get_the_loaders() 
construct_dcnn(args.model_name)
if not(args.pre_trained_model): train_dcnn_(50, train_loader) # just to assist the first training instance when not using pre-train 
print_all_results()

unlabeled_result, train_result, unlabeled_set, unlabeled_loader = semi_supervised_train(
        unlabeled_set, unlabeled_loader)
print_all_results()
#test_set = add_new_rand_labels(test_set)
#unlabeled_result, train_result, test_set, test_loader = semi_supervised_train(
#        test_set, test_loader)
# print('acc of using test with no labels', sum(test_set.labels == test_set.labels_original) / 8000)

if not(args.model_name_attention==''): 
    print('...... Train attention model ', args.model_name_attention, '......' )
    construct_dcnn(args.model_name_attention) # initialize the model, we can initialize just the 'lr'                  
    for i in range(4):
        args.lr= 0.01        
        train_dcnn_(80, unlabeled_loader)
        train_dcnn_(20, train_loader) # Now, slightly fine-tuninig with only the train set
print_all_results()
print("execution time: ", time.time()-start_timer ) 



