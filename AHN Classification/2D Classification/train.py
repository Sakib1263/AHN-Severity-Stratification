# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader
from torch.serialization import SourceChangeWarning
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
for warning in [UserWarning, SourceChangeWarning, Warning]:
    warnings.filterwarnings("ignore", category=warning)
# Data science tools
import os
import numpy as np
from os import path
from importlib import import_module
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# customized functions 
from utils import *
from models import *


# Parse command line arguments
fname = "config_train.py"
configuration = import_module(fname.split(".")[0])
config = configuration.config


if __name__ ==  '__main__':
    # torch.set_num_threads(1)
    ################## Network hyper-parameters 
    parentdir = config['parentdir']                     # root directory
    isPretrained = config['isPretrained']               # set to 'True' to use pretrained weights or set to 'False' to train from scratch
    model_mode = config['model_mode']                   # 'custom_CNN' | 'custom_ONN' | 'import_Torch' | 'import_TIMM'
    q_order = config['q_order']                         # qth order Maclaurin approximation, common values: {1,3,5,7,9}. q=1 is equivalent to CNN
    ONN = config['ONN']                                 # set to 'True' if you are using ONN
    input_ch = config['input_ch']                       # 1 for grayscale images, 3 for RGB images, 4 for RGBA images with an Alpha channel
    batch_size = config['batch_size']                   # Batch size, Change to fit hardware, common values: {4,8,16} for 2D datasets
    input_mean = config['input_mean']                   # Dataset mean
    input_std = config['input_std']                     # Dataset std
    loss_func = config['loss_func']                     # 'MSELoss', 'CrossEntropyLoss', etc. (https://pytorch.org/docs/stable/nn.html)
    optim_fc = config['optim_fc']                       # 'Adam', 'SGD', etc. (https://pytorch.org/docs/stable/optim.html)
    optim_scheduler = config['optim_scheduler']         # 'ReduceLROnPlateau', etc. (https://pytorch.org/docs/stable/optim.html)
    final_activation_func = config['final_activation_func']  # 'Sigmoid', 'Softmax', etc. (https://pytorch.org/docs/stable/nn.html)
    lr = config['lr']                                   # Learning rate
    stop_criteria = config['stop_criteria']             # Stopping criteria: 'loss' or 'Accuracy'
    n_epochs = config['n_epochs']                       # Number of training epochs
    epochs_patience = config['epochs_patience']         # If val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
    lr_factor = config['lr_factor']                     # Learning factor
    max_epochs_stop = config['max_epochs_stop']         # Maximum number of epochs with no improvement in validation metric for early stopping
    num_folds = config['num_folds']                     # Number of cross-validation folds
    Resize_h = config['Resize_h']                       # Network input (Image) height
    Resize_w = config['Resize_w']                       # Network input (Image) width
    load_weights = config['load_weights']               # Specify path of pretrained model weights or set to False to train from scratch
    model_to_load = config['model_to_load']             # Choose one of the models specified in config file
    model_name = config['model_name']                   # Choose a unique name for result folder
    aux_logits = config['aux_logits']                   # Required for models with auxilliary outputs (e.g., InceptionV3)
    fold_to_run = config['fold_to_run']                 # Define as [] to loop through all folds, or specify start and end folds i.e. [3 5]
    encoder = config['encoder']                         # Set to 'True' if you retrain Seg. model encoder as a classifer
    outdir = config['outdir']                         # The destination directory for saving the pipeline outputs (models, results, plots, etc.)
    
    if (model_mode == 'custom_CNN') or (model_mode == 'import_Torch') or (model_mode == 'import_TIMM'):
        ONN = False
        q_order = 1
    
    if ((model_mode == 'import_Torch') or (model_mode == 'import_TIMM')) and (isPretrained == True):
        input_ch = 3
    
    traindir = parentdir + 'Data/Train/'
    testdir =  parentdir + 'Data/Test/'
    valdir =  parentdir + 'Data/Val/'

    Results_path = outdir + 'Results/'
    # Create Results Directory 
    if path.exists(Results_path): 
        pass
    else:
        os.makedirs(Results_path)

    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on GPU: {train_on_gpu}')
    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} GPUs detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False 

    # loop through folds
    if not fold_to_run:
        loop_start = 1
        loop_end = num_folds + 1
    else:
        loop_start = fold_to_run[0]
        loop_end = fold_to_run[1] + 1

    for fold_idx in range(loop_start, loop_end):
        if fold_idx==loop_start:
            print('Training using '+ model_to_load +' network')
        print(f'Starting Fold {fold_idx}...')
        # Create Save Directory
        save_path = Results_path + f'/{model_name}/fold_{fold_idx}'
        if path.exists(save_path):
            pass
        else:
            os.makedirs(save_path) 
        save_file_name = save_path + '/' + model_name + f'_fold_{fold_idx}.pt'
        checkpoint_name = save_path + f'/checkpoint_{fold_idx}.pt'
        traindir_fold = traindir + f'fold_{fold_idx}/'
        testdir_fold = testdir + f'fold_{fold_idx}/' 
        valdir_fold = valdir + f'fold_{fold_idx}/' 
        
        # Image Dataloader
        # Image Transformation
        if ONN:
            if input_ch==3: 
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor()
                    ])
                my_test_transforms =  my_transforms  
            elif input_ch==1:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor()
                    ])
                my_test_transforms =  my_transforms
        else:
            if input_ch==1 and len(input_mean)==3: 
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std)  # gray
                    ])
                my_test_transforms =  my_transforms
            elif input_ch==1:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std)  # gray
                    ])
                my_test_transforms =  my_transforms
            else: 
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std)  # 3 channel
                    ])
                my_test_transforms =  my_transforms
        
        # Create train labels
        categories, n_Class_train, img_names_train, labels_train, class_to_idx, idx_to_class = Createlabels(traindir_fold)
        labels_train = torch.from_numpy(labels_train).to(torch.int64)
        class_num = len(categories)
        # Create val labels
        _, n_Class_val, img_names_val, labels_val, _, _ = Createlabels(valdir_fold)
        labels_val = torch.from_numpy(labels_val).to(torch.int64)
        # Create test labels
        _, n_Class_test, img_names_test, labels_test, _, _ = Createlabels(testdir_fold)
        labels_test = torch.from_numpy(labels_test).to(torch.int64)
        
        # Dataloaders
        # Train Dataloader 
        train_ds = MyData(root_dir=traindir_fold, categories=categories, img_names=img_names_train, target=labels_train, my_transforms=my_transforms, return_path=False, ONN=ONN, mean=input_mean, std=input_std)
        if (len(train_ds)/batch_size) == 0:
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1) 
        else:
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, drop_last=True)  
        # Validation Dataloader 
        val_ds = MyData(root_dir=valdir_fold, categories=categories, img_names=img_names_val, target=labels_val, my_transforms=my_test_transforms, return_path=False, ONN=ONN, mean=input_mean, std=input_std)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
        # Test Dataloader
        test_ds = MyData(root_dir=testdir_fold, categories=categories, img_names=img_names_test, target=labels_test, my_transforms=my_test_transforms, return_path=True, ONN=ONN, mean=input_mean, std=input_std)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)

        # Release Memeory (delete variables)
        del n_Class_train, img_names_train, labels_train
        del n_Class_val, img_names_val, labels_val 

        # Load Model
        if load_weights: 
            try:
                print('Loading previously trained model weights from local directory...')
                model = get_pretrained_model(parentdir, model_to_load, model_mode, isPretrained, input_ch, class_num, final_activation_func, train_on_gpu, multi_gpu, q_order) 
                checkpoint = torch.load(load_weights)
                model.load_state_dict(checkpoint['model_state_dict'])
                epochs_prev = checkpoint['epoch']
                print(f'Model had been trained previously for {epochs_prev} epochs.\n')
                if encoder:
                    model = EncoderModel(model.encoder)  # ResNet
                    model = model.to('cuda')
            except:
                raise ValueError("The shape of the loaded weights do not exactly match with the model framework.")
        else: 
            model = get_pretrained_model(parentdir, model_to_load, model_mode, isPretrained, input_ch, class_num, final_activation_func, train_on_gpu, multi_gpu, q_order) 
            epochs_prev = 0
        # Check if model on cuda
        if next(model.parameters()).is_cuda:
            print('Model device: CUDA')
            print('==============================================================')
        
        # Loss Function
        if loss_func == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
        elif loss_func == 'NLLLoss':
            criterion = nn.NLLLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean')
        elif loss_func == 'MultiMarginLoss':
            criterion = nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
        else:
            raise ValueError('Choose a valid loss function from here: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer')
        
        if train_on_gpu:
            criterion = criterion.to('cuda')
        
        # Optimizer
        if optim_fc == 'Adagrad':  
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, foreach=None, maximize=False)
        elif optim_fc == 'Adam':  
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=False)
        elif optim_fc == 'AdamW':  
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False, foreach=None, capturable=False)
        elif optim_fc == 'Adamax':  
            optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None, maximize=False)
        elif optim_fc == 'NAdam':  
            optimizer = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, foreach=None)
        elif optim_fc == 'RAdam':  
            optimizer = torch.optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None)
        elif optim_fc == 'RMSprop':  
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)
        elif optim_fc == 'Rprop':  
            optimizer = torch.optim.Rprop(model.parameters(), lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50), foreach=None, maximize=False)
        elif optim_fc == 'SGD': 
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False, maximize=False, foreach=None, differentiable=False)
        else:
            raise ValueError('The pipeline does not support this optimizer. Choose a valid optimizer function from here: https://pytorch.org/docs/stable/optim.html')

        # Training Scheduler
        if optim_scheduler == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=epochs_patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        else:
            raise ValueError('The pipeline does not support this as the scheduler. Choose a valid scheduler from here: https://pytorch.org/docs/stable/optim.html')
        
        # Training
        model, history = train(model_to_load, model, stop_criteria, criterion, optimizer, scheduler, train_dl, val_dl, test_dl, 
        checkpoint_name, train_on_gpu, aux_logits, history=[], max_epochs_stop=max_epochs_stop, n_epochs=n_epochs, epochs_prev=epochs_prev, print_every=1)

        # Saving TrainModel
        TrainChPoint = {} 
        TrainChPoint['model'] = model                              
        TrainChPoint['history'] = history
        TrainChPoint['categories'] = categories
        TrainChPoint['class_to_idx'] = class_to_idx
        TrainChPoint['idx_to_class'] = idx_to_class
        torch.save(TrainChPoint, save_file_name) 

        # Training Results
        # We can inspect the training progress by looking at the `history`. 
        # plot loss
        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'val_loss', 'test_loss']:
            plt.plot(history[c], label=c) 
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(save_path+f'/LossPerEpoch_fold_{fold_idx}.png')
        # plt.show()
        # plot accuracy
        plt.figure(figsize=(8, 6))
        for c in ['train_acc', 'val_acc', 'test_acc']:
            plt.plot(100 * history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch') 
        plt.ylabel('Accuracy') 
        plt.savefig(save_path+f'/AccuracyPerEpoch_fold_{fold_idx}.png')
        # plt.show()

        # # Create test labels
        # _, n_Class_test, img_names_test, labels_test, _, _ = Createlabels(testdir_fold)   
        # labels_test = torch.from_numpy(labels_test).to(torch.int64) 
        # # test dataloader
        # test_ds = MyData(root_dir=testdir_fold,categories=categories,img_names=img_names_test,target=labels_test,my_transforms=my_transforms,return_path=True,ONN=ONN,mean=input_mean,std=input_std)
        # test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)

        # release memeory (delete variables)
        del my_transforms, optimizer, scheduler
        del train_ds, train_dl, val_ds, val_dl
        del img_names_test, labels_test 
        del TrainChPoint
        torch.cuda.empty_cache()
        # Test Accuracy
        all_paths =list()
        test_acc = 0.0
        test_loss = 0.0
        i=0
        model.eval() 
        for data, targets, im_path in test_dl:
            # Tensors to gpu
            if train_on_gpu:
                data, targets = data.to('cuda', non_blocking=True), targets.to('cuda', non_blocking=True)
            # all_targets = torch.cat([all_targets ,targets.numpy()])
            # Raw model output
            out = model(data)
            if aux_logits == False or model_to_load == 'inception_v3':
                loss = criterion(out, targets)
                # loss = criterion(output, to_one_hot(target).to('cuda'))  # use it with mse loss 
            elif aux_logits == True and model_to_load != 'inception_v3':
                loss = 0
                for k in range(0,len(out)):
                    loss += (criterion(out[k], targets))/(2**k)
                    # loss += criterion(output[k], to_one_hot(target).to('cuda'))  # use it with mse loss 
            test_loss += loss.item() * data.size(0)
            if aux_logits == False or model_to_load == 'inception_v3':
                output = torch.exp(out)
            elif aux_logits == True and model_to_load != 'inception_v3':
                output = torch.exp(out[0])
            # pred_probs = torch.cat([pred_probs, output])
            all_paths.extend(im_path)
            targets = targets.cpu()
            if i==0:
                all_targets = targets.numpy()
                pred_probs = output.cpu().detach().numpy()
            else:
                all_targets = np.concatenate((all_targets, targets.numpy()))
                pred_probs = np.concatenate((pred_probs, output.cpu().detach().numpy()))
            _, temp_label = torch.max(output.cpu(), dim=1)
            correct_tensor = temp_label.eq(targets.data.view_as(temp_label))      # this lin is temporary 
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))   # this lin is temporary 
            test_acc += accuracy.item() * data.size(0)                      # this lin is temporary 
            temp_label = temp_label.detach().numpy()
            if i==0:
                pred_label = temp_label
            else:
                pred_label = np.concatenate((pred_label  ,temp_label))
            i +=1
        test_loss = test_loss / len(test_dl.dataset)
        test_loss = round(test_loss,4)
        test_acc = test_acc / len(test_dl.dataset)                          # this lin is temporary
        test_acc = round(test_acc*100,2)
        print(f'Test Loss: {test_loss},  Test Accuracy: {test_acc}%')
        from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
        # main confusion matrix
        cm = confusion_matrix(all_targets, pred_label)
        # it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
        # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP   
        cm_per_class = multilabel_confusion_matrix(all_targets, pred_label)
        # Saving Test Results
        save_file_name = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
        TestChPoint = {} 
        TestChPoint['categories']=categories
        TestChPoint['class_to_idx']=class_to_idx
        TestChPoint['idx_to_class']=idx_to_class
        TestChPoint['Train_history']=history 
        TestChPoint['n_Class_test']=n_Class_test
        TestChPoint['targets']=all_targets
        TestChPoint['prediction_label']=pred_label
        TestChPoint['prediction_probs']=pred_probs
        TestChPoint['image_names']=all_paths 
        TestChPoint['cm']=cm
        TestChPoint['cm_per_class']=cm_per_class
        torch.save(TestChPoint, save_file_name)
        # torch.load(save_file_name)
        if fold_idx == loop_start:
            cumulative_cm = cm
        else:
            cumulative_cm += cm
        # Release memeory (delete variables)
        del model, criterion, history, test_ds, test_dl
        del data, targets, out, output, temp_label, 
        del test_acc, test_loss, loss
        del pred_probs, pred_label, all_targets, all_paths, 
        del cm, cm_per_class, TestChPoint
        torch.cuda.empty_cache()
        # Delete checkpoint 
        # os.remove(checkpoint_name)
        # print("Checkpoint File Removed!")
        print(f'Completed fold {fold_idx}')
    print('==============================================================')
    Overall_Accuracy = np.sum(np.diagonal(cumulative_cm)) / np.sum(cumulative_cm)
    Overall_Accuracy = round(Overall_Accuracy*100, 2)
    print('Cummulative Confusion Matrix')
    print(cumulative_cm)
    print(f'Overall Test Accuracy: {Overall_Accuracy}')
    print('==============================================================')

