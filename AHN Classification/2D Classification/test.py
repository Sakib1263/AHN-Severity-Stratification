# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
from torchvision import transforms
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader 
import torch.nn as nn
# Warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
from PIL import Image
from importlib import import_module
from scipy.io import loadmat, savemat
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
# Timing utility
from timeit import default_timer as timer
# Custom functions 
from utils import *
from models import *


def Generate_CSV(model, Data_Loader, save_path, idx_to_class, aux_logits, dataset='Train'):
    # Generate Train CSV 
    all_paths =list()
    i=0
    for data, targets, im_path in Data_Loader:
        # Tensors to gpu
        if train_on_gpu:
            data = data.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking=True)
        out = model(data)
        if aux_logits == False:
            output = torch.exp(out)
        elif aux_logits == True:
            output = torch.exp(out[0])
        # images names
        all_paths.extend(im_path)
        targets = targets.cpu()
        if i==0:
            all_targets = targets.numpy()
            pred_probs = output.cpu().detach().numpy()
        else:
            all_targets = np.concatenate((all_targets, targets.numpy()))
            pred_probs = np.concatenate((pred_probs, output.cpu().detach().numpy()))
        _, temp_label = torch.max(output.cpu(), dim=1)
        temp_label = temp_label.detach().numpy()
        if i==0:
            pred_label = temp_label
        else:
            pred_label = np.concatenate((pred_label, temp_label))
        i +=1
    # Generate CSV from pandas dataframe
    output_csv = pd.DataFrame([])
    output_csv['image_name'] = all_paths
    output_csv['target'] = all_targets
    output_csv['pred'] = pred_label
    for prob_column in range(pred_probs.shape[1]):
        output_csv[idx_to_class[prob_column]] = pred_probs[:,prob_column]
    output_csv.to_csv(save_path + f'/{model_name}_output_{dataset}_prob_fold_{fold_idx}.csv')
    print(f'{dataset} Dataframe Write to CSV - Done')
    del model
    del data, targets, out, output, temp_label  
    del pred_probs, pred_label, all_targets, all_paths


def Generate_UnlabeledData_CSV(model, Data_Loader, save_path, idx_to_class, aux_logits, dataset='Train'):
    all_paths =list()
    i=0
    for data, _, im_path in Data_Loader: 
        # Tensors to gpu
        if train_on_gpu:
            data = data.to('cuda', non_blocking=True)
        # model output
        out = model(data)
        if aux_logits == False:
            output = torch.exp(out)
        elif aux_logits == True:
            output = torch.exp(out[0])
        # images names
        all_paths.extend(im_path)
        if i==0:
            pred_probs = output.cpu().detach().numpy()
        else:
            pred_probs = np.concatenate((pred_probs, output.cpu().detach().numpy()))
        _, temp_label = torch.max(output.cpu(), dim=1)
        temp_label = temp_label.detach().numpy()
        if i==0:
            pred_label = temp_label
        else:
            pred_label = np.concatenate((pred_label, temp_label))
        i +=1

    output_csv = pd.DataFrame([])
    output_csv['image_name'] = all_paths
    output_csv['pred'] = pred_label
    
    for prob_column in range(pred_probs.shape[1]):
        output_csv[idx_to_class[prob_column]] = pred_probs[:,prob_column]
    output_csv.to_csv(save_path + f'/{model_name}_output_{dataset}_prob_fold_{fold_idx}.csv')
    print(f'{dataset} Dataframe Write to CSV - Done')
    del model
    del data, targets, out, output, temp_label  
    del pred_probs, pred_label, all_paths


# Parse command line arguments
fname = "config_test.py" 
configuration = import_module(fname.split(".")[0])
config = configuration.config

if __name__ ==  '__main__':  
    # torch.set_num_threads(1)
    # Network hyper-parameters 
    parentdir = config['parentdir']                     # main directory
    ONN = config['ONN']                                 # set to 'True' if you are using ONN
    input_ch = config['input_ch']                       # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays
    batch_size = config['batch_size']                   # batch size, Change to fit hardware
    num_folds = config['num_folds']                     # number of cross validation folds
    CI = config['CI']                                   # Confidence interval (missied cases with probability>=CI will be reported in excel file)
    input_mean = config['input_mean']                   # dataset mean per channel
    input_std = config['input_std']                     # dataset std per channel
    loss_func = config['loss_func']                     # 'MSELoss', 'CrossEntropyLoss', etc. (https://pytorch.org/docs/stable/nn.html)
    Resize_h = config['Resize_h']                       # network input size
    Resize_w = config['Resize_w']  
    load_weights = config['load_weights']               # specify full path of pretrained model pt file or set to False to load pretrained model based on model name and fold num
    labeled_Data =  config['labeled_Data']              # set to true if you have the labeled test set
    aux_logits = config['aux_logits']                   # Required for models with auxilliary outputs (e.g., InceptionV3)
    model_name = config['model_name']                   # name of trained model .pt file
    fold_to_run = config['fold_to_run']                 # define as [] to loop through all folds, or specify start and end folds i.e. [3 5]
    N_steps = config['N_steps']                         # Number of steps for inference
    outdir = config['outdir']                           # The destination directory for saving the pipeline outputs (models, results, plots, etc.)
    
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
    test_history = []
    index = []
    # loop through folds
    if not fold_to_run:
        loop_start = 1
        loop_end = num_folds + 1
    else:
        loop_start = fold_to_run[0]
        loop_end = fold_to_run[1] + 1
    for fold_idx in range(loop_start, loop_end):
        print('==============================================================')
        print(f'Starting Fold {fold_idx}...')
        # Create Save Directory
        save_path = Results_path + f'/{model_name}/fold_{fold_idx}'
        if path.exists(save_path):
            pass
        else:
            os.makedirs(save_path) 
        traindir_fold = traindir + f'fold_{fold_idx}/' 
        valdir_fold = valdir + f'fold_{fold_idx}/' 
        testdir_fold = testdir + f'fold_{fold_idx}/' 
        # load model
        if load_weights == False: 
            print('Loading previously trained model checkpoint...')
            pt_file = save_path + f'/{model_name}_fold_{fold_idx}.pt'
            checkpoint = torch.load(pt_file)
            model = checkpoint['model'] 
            categories = checkpoint['categories']
            class_to_idx = checkpoint['class_to_idx']
            idx_to_class = checkpoint['idx_to_class'] 
            del pt_file, checkpoint
        else:
            print('Loading previously trained model...')
            checkpoint = torch.load(load_weights)
            model = checkpoint['model'] 
            categories = checkpoint['categories']
            class_to_idx = checkpoint['class_to_idx']
            idx_to_class = checkpoint['idx_to_class'] 
            del checkpoint 
        model.eval()
        model = model.to('cuda')
        # check if model on cuda
        if next(model.parameters()).is_cuda:
            print('Model Device: cuda')
        # Loss Function
        if loss_func == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
        elif loss_func == 'NLLLoss':
            criterion = nn.NLLLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean')
        elif loss_func == 'MultiMarginLoss':
            criterion = nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
        else:
            raise ValueError('Choose a valid loss function from here: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer')
        # Image Transformation
        if ONN:
            if input_ch==3: 
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor()
                    ])   
            elif input_ch==1:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor()
                    ]) 
        else:
            if input_ch==1 and len(input_mean)==3: 
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std)  # gray
                    ])   
            elif input_ch==1:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std)  # gray
                    ]) 
            else:  
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC), 
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=input_mean, std=input_std)  # 3 channel
                    ])

        # Dataloaders
        # Create train labels
        categories, n_Class_train, img_names_train, labels_train, class_to_idx, idx_to_class = Createlabels(traindir_fold)
        labels_train = torch.from_numpy(labels_train).to(torch.int64)
        class_num = len(categories)
        # Train dataloader
        train_ds = MyData(root_dir=traindir_fold, categories=categories, img_names=img_names_train, target=labels_train, my_transforms=my_transforms, return_path=True, ONN=ONN, mean=input_mean, std=input_std)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
        # Create val labels
        _, n_Class_val, img_names_val, labels_val, _, _ = Createlabels(valdir_fold)
        labels_val = torch.from_numpy(labels_val).to(torch.int64)
        # Val dataloader
        val_ds = MyData(root_dir=valdir_fold, categories=categories, img_names=img_names_val, target=labels_val, my_transforms=my_transforms, return_path=True, ONN=ONN, mean=input_mean, std=input_std)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
        # Create test labels
        _, n_Class_test, img_names_test, labels_test, _, _ = Createlabels(testdir_fold)  
        labels_test = torch.from_numpy(labels_test).to(torch.int64)
        # Test Dataloader
        test_ds = MyData(root_dir=testdir_fold, categories=categories, img_names=img_names_test, target=labels_test, my_transforms=my_transforms, return_path=True, ONN=ONN, mean=input_mean, std=input_std)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
        print(f'Combined Evaluation of Folds {loop_start} to {loop_end-1}...') 
        if labeled_Data:
            all_paths = list()
            test_acc = 0.0
            test_loss = 0.0
            pbar = tqdm(test_dl, desc=f"Testing")
            for ii, (data, targets, im_path) in enumerate(pbar):
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                    targets = targets.to('cuda', non_blocking=True)
                # model output
                out = model(data)
                # loss
                if aux_logits == False:
                    loss = criterion(out, targets)
                    # loss = criterion(output, to_one_hot(target).to('cuda'))  # use it with mse loss 
                elif aux_logits == True:
                    loss = 0
                    for k in range(0,len(out)):
                        loss += (criterion(out[k], targets))/(2**k)
                        # loss += criterion(output[k], to_one_hot(target).to('cuda'))  # use it with mse loss 
                test_loss += loss.item() * data.size(0)
                if aux_logits == False:
                    output = torch.exp(out)
                elif aux_logits == True:
                    output = torch.exp(out[0])
                # images names
                all_paths.extend(im_path)
                targets = targets.cpu()
                if ii==0:
                    all_targets = targets.numpy()
                    pred_probs = output.cpu().detach().numpy()
                else:
                    all_targets = np.concatenate((all_targets, targets.numpy()))
                    pred_probs = np.concatenate((pred_probs, output.cpu().detach().numpy()))
                _, temp_label = torch.max(output.cpu(), dim=1)
                correct_tensor = temp_label.eq(targets.data.view_as(temp_label))      # this line is temporary 
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))         # this line is temporary 
                test_acc += accuracy.item() * data.size(0)                            # this line is temporary 
                temp_label = temp_label.detach().numpy()
                if ii==0:
                    pred_label = temp_label
                else:
                    pred_label = np.concatenate((pred_label,temp_label))
            test_loss = test_loss / len(test_dl.dataset) 
            test_loss = round(test_loss,4)
            test_acc = test_acc / len(test_dl.dataset)                          # this lin is temporary
            test_acc = round(test_acc*100,2)
            print(f'Test Loss: {test_loss},  Test Accuracy: {test_acc}%')
            # main confusion matrix
            cm = confusion_matrix(all_targets, pred_label)
            # it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
            # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP   
            cm_per_class = multilabel_confusion_matrix(all_targets, pred_label)
            print('Saving Test Results...') 
            # Saving Test Results
            save_file_name = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
            TestChPoint = {} 
            TestChPoint['categories']=categories
            TestChPoint['class_to_idx']=class_to_idx
            TestChPoint['idx_to_class']=idx_to_class
            TestChPoint['n_Class_test']=n_Class_test
            TestChPoint['targets']=all_targets
            TestChPoint['prediction_label']=pred_label
            TestChPoint['prediction_probs']=pred_probs
            TestChPoint['image_names']=all_paths 
            TestChPoint['cm']=cm
            TestChPoint['cm_per_class']=cm_per_class
            torch.save(TestChPoint, save_file_name)
            print('Generating CSV Files from Individual Predictions...')
            # Generate Train CSV
            # Generate_CSV(model=model, Data_Loader=train_dl, save_path=save_path, idx_to_class=idx_to_class, aux_logits=aux_logits, dataset='Train')
            # Generate Val CSV
            # Generate_CSV(model=model, Data_Loader=val_dl, save_path=save_path, idx_to_class=idx_to_class, aux_logits=aux_logits, dataset='Val')
            # Generate Test CSV
            # Generate_CSV(model=model, Data_Loader=test_dl, save_path=save_path, idx_to_class=idx_to_class, aux_logits=aux_logits, dataset='Test')
        else:
            all_paths =list()
            pbar = tqdm(test_dl, desc=f"Testing")
            for ii, (data, _, im_path) in enumerate(pbar): 
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                # model output
                out = model(data)
                if aux_logits == False:
                    output = torch.exp(out)
                elif aux_logits == True:
                    output = torch.exp(out[0])
                # images names
                all_paths.extend(im_path)
                if ii==0:
                    pred_probs = output.cpu().detach().numpy()
                else:
                    pred_probs = np.concatenate((pred_probs, output.cpu().detach().numpy()))
                _, temp_label = torch.max(output.cpu(), dim=1)
                temp_label = temp_label.detach().numpy()
                if ii==0:
                    pred_label = temp_label
                else:
                    pred_label = np.concatenate((pred_label, temp_label))
            print('Saving Test Results...') 
            # Saving Test Results
            save_file_name = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
            TestChPoint = {} 
            TestChPoint['categories']=categories
            TestChPoint['class_to_idx']=class_to_idx
            TestChPoint['idx_to_class']=idx_to_class
            TestChPoint['n_Class_test']=n_Class_test 
            TestChPoint['prediction_label']=pred_label
            TestChPoint['prediction_probs']=pred_probs
            TestChPoint['image_names']=all_paths 
            torch.save(TestChPoint, save_file_name) 
            print('Generating CSV Files from Individual Predictions...') 
            # Generate Train CSV
            # Generate_UnlabeledData_CSV(model=model, Data_Loader=train_dl, save_path=save_path, idx_to_class=idx_to_class, aux_logits=aux_logits, dataset='Train')
            # Generate Val CSV
            # Generate_UnlabeledData_CSV(model=model, Data_Loader=val_dl, save_path=save_path, idx_to_class=idx_to_class, aux_logits=aux_logits, dataset='Val')
            # Generate Test CSV
            # Generate_UnlabeledData_CSV(model=model, Data_Loader=test_dl, save_path=save_path, idx_to_class=idx_to_class, aux_logits=aux_logits, dataset='Test')
        # Measure Inference Time
        Total_time = 0.0
        for i in range(N_steps):
            input_time = timer()
            out = model(data) 
            output_time = timer() 
            output_time = output_time - input_time
            Total_time = Total_time + output_time
            del out
        Total_time = Total_time/N_steps
        print(f'Total Inference Time: {Total_time*1000:.2} ms') 
        
        # Release memeory (delete variables)
        if labeled_Data: 
            del model, criterion, test_ds, test_dl
            del data, targets, output, temp_label  
            del test_acc, test_loss, loss
            del pred_probs, pred_label, all_targets, all_paths, 
            del cm, cm_per_class, TestChPoint
        else:
            del model, criterion, test_ds, test_dl
            del data, out, output, temp_label  
            del pred_probs, pred_label, all_paths, 
            del TestChPoint 
        torch.cuda.empty_cache()
        print(f'Completed fold {fold_idx}') 
    print('==============================================================')
    # Combined Evaluation of All CV Folds
    save_path = Results_path + '/' + model_name
    if labeled_Data: 
        all_Missed_c = list()
        for fold_idx in tqdm(range(loop_start, loop_end), desc="Combined Evaluation"):
            # Load checkpoint
            fold_path = save_path + f'/fold_{fold_idx}'
            model_path = fold_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
            TestChPoint = torch.load(model_path)
            if fold_idx==loop_start:
                targets = TestChPoint['targets']
                pred = TestChPoint['prediction_label']
                pred_probs = TestChPoint['prediction_probs']
                image_names  = TestChPoint['image_names']
            else: 
                targets = np.concatenate([targets, TestChPoint['targets']])
                pred = np.concatenate([pred, TestChPoint['prediction_label']])
                pred_probs = np.concatenate([pred_probs, TestChPoint['prediction_probs']])
                image_names.extend(TestChPoint['image_names'])
            # find missed cases (probs, image path)
            categories = TestChPoint['categories']
            n = len(categories)
            current_fold_target = TestChPoint['targets']
            current_fold_pred = TestChPoint['prediction_label']
            current_fold_image_names = TestChPoint['image_names'] 
            current_fold_prediction_probs = TestChPoint['prediction_probs'] 
            missed_idx = np.argwhere(1*(current_fold_target==current_fold_pred) == 0)
            m = len(missed_idx)
            missed_probs = np.zeros((m,n)) 
            for i in range(len(missed_idx)):
                index = int(missed_idx[i])
                all_Missed_c.extend([f'fold_{fold_idx}/'+current_fold_image_names[index]])
                missed_probs[i,:] = current_fold_prediction_probs[index,:] 
            if fold_idx==loop_start:
                all_missed_p = missed_probs
            else: 
                all_missed_p = np.concatenate((all_missed_p, missed_probs))
            # main confusion matrix
            cm = confusion_matrix(current_fold_target, current_fold_pred)
            # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
            cm_per_class = multilabel_confusion_matrix(current_fold_target, current_fold_pred)
            # Overall Accuracy
            Overall_Accuracy = np.sum(np.diagonal(cm))/np.sum(cm)
            Overall_Accuracy = round(Overall_Accuracy*100, 2)
            # Create confusion matrix table (pd.DataFrame)
            cm_table = pd.DataFrame(cm, index=categories, columns=categories)
            # Generate Confusion matrix figure
            cm_plot = plot_conf_mat(current_fold_target, current_fold_pred, labels=categories)
            cm_plot.savefig(fold_path + f'/{model_name}_Confusion_Matrix_fold_{fold_idx}.png', dpi=600)
            # Generate Multiclass ROC curve figure
            # roc_plot = plot_multiclass_roc(current_fold_target, current_fold_pred, categories)
            # roc_plot.savefig(fold_path + f'/{model_name}_ROC_plot_fold_{fold_idx}.png', dpi=600)
            # Generate Multiclass precision-recall curve figure
            # prc_plot = plot_multiclass_precision_recall_curves(current_fold_target, current_fold_pred, categories)
            # prc_plot.savefig(fold_path + f'/{model_name}_PRC_plot_fold_{fold_idx}.png', dpi=600)
            Eval_Mat = []
            # Per class metricies
            for i in range(len(categories)):
                TN = cm_per_class[i][0][0]
                FP = cm_per_class[i][0][1]
                FN = cm_per_class[i][1][0]
                TP = cm_per_class[i][1][1]
                Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                Precision = round(100*(TP)/(TP+FP), 2)  
                Sensitivity = round(100*(TP)/(TP+FN), 2) 
                F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)  
                Specificity = round(100*(TN)/(TN+FP), 2)  
                Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
            # Sizes of each class
            s = np.sum(cm,axis=1) 
            # Create tmep excel table 
            headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
            temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
            # Weighted average of per class metricies
            Accuracy = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2) 
            Precision = round(temp_table['Precision'].dot(s)/np.sum(s), 2)  
            Sensitivity = round(temp_table['Sensitivity'].dot(s)/np.sum(s), 2)  
            F1_score = round(temp_table['F1_score'].dot(s)/np.sum(s), 2)  
            Specificity = round(temp_table['Specificity'].dot(s)/np.sum(s), 2)   
            values = [Accuracy, Precision, Sensitivity, F1_score, Specificity]
            # Create per class metricies excel table with weighted average row
            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
            categories_wa = categories + ['Weighted Average']
            Eval_table = pd.DataFrame(Eval_Mat, index=categories_wa, columns=headers)
            # Create confusion matrix table (pd.DataFrame)
            Overall_Acc = pd.DataFrame(Overall_Accuracy, index=['Overall_Accuracy'] , columns=[' '])
            # Save to excel file   
            new_savepath = fold_path + f'/{model_name}_fold_{fold_idx}.xlsx'  # file to save 
            writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
            # Sheet 1 (Evaluation metricies) + (Commulative Confusion Matrix) 
            col = 1; row = 1 
            Eval_table.to_excel(writer, "Results", startcol=col, startrow=row)
            row = row+7+len(class_to_idx)
            Overall_Acc.to_excel(writer, "Results", startcol=col, startrow=row, header=None)
            col = col+8; row=1   
            Predicted_Class = pd.DataFrame(['Predicted Class'])
            Predicted_Class.to_excel(writer, "Results", startcol=col+1, startrow=row, header=None, index=None)
            row = 2     
            cm_table.to_excel(writer, "Results", startcol=col, startrow=row)
            # save 
            writer.close()
        # find missed cases with high CI (probs, image path)
        temp = np.max(all_missed_p, axis=1)
        temp_idx = np.argwhere(temp >= CI) 
        unsure_missed_c = list() 
        unsure_missed_p = np.zeros((len(temp_idx), n)) 
        for i in range(len(temp_idx)):
            index = int(temp_idx[i])
            unsure_missed_c.extend([ all_Missed_c[index] ]) 
            unsure_missed_p[i,:] =  all_missed_p[index,:]
        categories = TestChPoint['categories']
        n = len(categories)
        from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
        # main confusion matrix
        class_index  = list(range(0,class_num))
        cm = confusion_matrix(y_true=targets, y_pred=pred, labels=class_index)
        # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
        cm_per_class = multilabel_confusion_matrix(targets, pred)
        # Overall Accuracy
        Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
        Overall_Accuracy = round(Overall_Accuracy*100, 2)
        # create missed and unsure missed tables
        missed_table = pd.DataFrame(all_Missed_c, columns=[f'Missed Cases'])
        unsure_table = pd.DataFrame(unsure_missed_c, columns=[f'Unsure Missed Cases (CI={CI})'])
        missed_prob_table = pd.DataFrame(np.round(all_missed_p,4), columns=categories) 
        unsure_prob_table = pd.DataFrame(np.round(unsure_missed_p,4), columns=categories) 
        # Create confusion matrix table (pd.DataFrame)
        cm_table = pd.DataFrame(cm, index=categories, columns=categories)
        # Generate Confusion matrix figure
        cm_plot = plot_conf_mat(targets, pred, labels=categories)
        cm_plot.savefig(Results_path + f'/{model_name}/{model_name}_Overall_Confusion_Matrix.png', dpi=600)
        # Generate Multiclass ROC curve figure
        # roc_plot = plot_multiclass_roc(targets, pred_probs, categories)
        # roc_plot.savefig(Results_path + f'/{model_name}/{model_name}_Overall_ROC_plot.png', dpi=600)
        # Generate Multiclass precision-recall curve figure
        # prc_plot = plot_multiclass_precision_recall_curves(targets, pred_probs, categories)
        # prc_plot.savefig(Results_path + f'/{model_name}/{model_name}_Overall_PRC_plot.png', dpi=600)
        Eval_Mat = []
        # Per class metricies
        for i in range(len(categories)):
            TN = cm_per_class[i][0][0] 
            FP = cm_per_class[i][0][1]   
            FN = cm_per_class[i][1][0]  
            TP = cm_per_class[i][1][1]  
            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
            Precision = round(100*(TP)/(TP+FP), 2)  
            Sensitivity = round(100*(TP)/(TP+FN), 2) 
            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)  
            Specificity = round(100*(TN)/(TN+FP), 2)  
            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
        # Sizes of each class
        s = np.sum(cm,axis=1) 
        # Create tmep excel table 
        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
        # Weighted average of per class metricies
        Accuracy = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2) 
        Precision = round(temp_table['Precision'].dot(s)/np.sum(s), 2)  
        Sensitivity = round(temp_table['Sensitivity'].dot(s)/np.sum(s), 2)  
        F1_score = round(temp_table['F1_score'].dot(s)/np.sum(s), 2)  
        Specificity = round(temp_table['Specificity'].dot(s)/np.sum(s), 2)   
        values = [Accuracy, Precision, Sensitivity, F1_score, Specificity]
        # Create per class metricies excel table with weighted average row
        Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
        categories_wa = categories + ['Weighted Average']
        Eval_table = pd.DataFrame(Eval_Mat, index=categories_wa, columns=headers)
        # Create confusion matrix table (pd.DataFrame)
        Overall_Acc = pd.DataFrame(Overall_Accuracy, index=['Overall_Accuracy'] , columns=[' '])
        print('\n') 
        print('Cumulative Confusion Matrix')
        print('---------------------------')
        print(cm_table) 
        print('\n') 
        print('Evaluation Matrices (Overall)')
        print('-------------------')
        print(Eval_table)
        print(Overall_Acc)
        # Save to excel file   
        new_savepath = Results_path + f'/{model_name}/{model_name}_Overall_Outcomes.xlsx'  # file to save 
        writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
        # Sheet 1 (Unsure missed cases) + (Evaluation metricies) + (Commulative Confusion Matrix) 
        col = 0; row = 2 
        unsure_table.to_excel(writer, "Results", startcol=col,startrow=row) 
        col = 2; row = 2 
        unsure_prob_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
        col = col+n+2; row = 2 
        Eval_table.to_excel(writer, "Results", startcol=col,startrow=row)
        row = row+7+len(class_to_idx)
        Overall_Acc.to_excel(writer, "Results", startcol=col,startrow=row, header=None)
        col = col+8; row=1   
        Predicted_Class = pd.DataFrame(['Predicted Class'])
        Predicted_Class.to_excel(writer, "Results", startcol=col+1,startrow=row, header=None, index=None)
        row = 2     
        cm_table.to_excel(writer, "Results", startcol=col,startrow=row)
        # Sheet 2 (All missed cases)
        col = 0; row = 2 
        missed_table.to_excel(writer, "Extra", startcol=col,startrow=row)
        col = 2; row = 2 
        missed_prob_table.to_excel(writer, "Extra", startcol=col,startrow=row, index=None)
        # save 
        writer.close()  
        # Save needed variables to create ROC curves 
        ROC_checkpoint = {} 
        ROC_checkpoint['prediction_label'] = pred
        ROC_checkpoint['prediction_probs'] = pred_probs
        ROC_checkpoint['targets'] = targets
        ROC_checkpoint['class_to_idx']=class_to_idx
        ROC_checkpoint['idx_to_class']=idx_to_class 
        ROC_path_pt = Results_path +'/'+  model_name + f'/{model_name}_overall_roc_inputs.pt'  # file to save 
        ROC_path_mat = Results_path +'/'+  model_name + f'/{model_name}_overall_roc_inputs.mat'  # file to save 
        torch.save(ROC_checkpoint,ROC_path_pt) 
        savemat(ROC_path_mat, ROC_checkpoint) 
    else:
        for fold_idx in tqdm(range(loop_start, loop_end), desc="Combined Evaluation"):
            # load checkpoint 
            fold_path = save_path + f'/fold_{fold_idx}'
            model_path = fold_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
            TestChPoint = torch.load(model_path)
            categories = TestChPoint['categories']  
            temp_pred = TestChPoint['prediction_label'] 
            pred_probs = TestChPoint['prediction_probs'] 
            image_names  = TestChPoint['image_names']
            for i in range(len(temp_pred)): 
                if i==0:
                    pred = [ idx_to_class[temp_pred[i]] ] 
                else:
                    pred.extend([ idx_to_class[temp_pred[i]] ])  
            # create missed and unsure missed tables
            input_names_table = pd.DataFrame(image_names, columns=[f'Input Image']) 
            pred_table = pd.DataFrame(pred, columns=[f'Prediction']) 
            prob_table = pd.DataFrame(np.round(pred_probs,4), columns=categories) 
            # save to excel file   
            new_savepath = fold_path + f'/{model_name}_fold_{fold_idx}.xlsx'  # file to save 
            writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
            # sheet 1 (input images) + (predictions) + (predictions probabilities) 
            col =0; row =2 
            input_names_table.to_excel(writer, "Results", startcol=col,startrow=row)
            col =2; row =2  
            pred_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
            col =3; row =2 
            prob_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
            # save 
            writer.close()
    print('==============================================================')
