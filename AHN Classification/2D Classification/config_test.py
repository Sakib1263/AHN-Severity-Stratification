# CNN test configuration file
config = {}
config['parentdir'] = ''                    # Root directory
# Set to 'True' if you are using ONN
config['ONN'] = False
# 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays
config['input_ch'] = 3
# Batch size, Change to fit hardware
config['batch_size'] = 4
# Number of cross-validation folds
config['num_folds'] = 1
# Confidence interval (missied cases with probability>=CI will be reported in excel file)
config['CI'] = 0.9
# config['input_mean'] = [0.2936]                     # Dataset mean per channel, b/w [0,1]
# config['input_std'] = [0.1895]                      # Dataset std per channel,  b/w [0,1]
# Dataset mean per channel, RGB or RGBA [0,1]
config['input_mean'] = [0.0609,0.0609,0.0609]
# Dataset std per channel,  RGB or RGBA [0,1]
config['input_std'] = [0.1540,0.1540,0.1540]
# 'MSELoss', 'CrossEntropyLoss', etc. (https://pytorch.org/docs/stable/nn.html)
config['loss_func'] = 'CrossEntropyLoss'
# Network input (Image) height
config['Resize_h'] = 224
# Network input (Image) width
config['Resize_w'] = config['Resize_h']
# config['load_weights'] ='/content/gdrive/MyDrive/EthnicData/Results/mobilenet_v2/mobilenet_v2_fold_1.pt'    # specify full path of pretrained model pt file
# Specify path of pretrained model wieghts or set to False to train from scratch
config['load_weights'] = False
# Set to true if you have the labeled test set
config['labeled_Data'] = True
# Required for models with auxilliary outputs (now, only for some custom_CNN models)
config['aux_logits'] = False
# Name of trained model .pt file, same name used in train code
config['model_name'] = 'convnext_xlarge_AHN_Classification'
# Specify a new folder name to save test results
config['new_name'] = 'convnext_xlarge_AHN_Classification'
# Number of steps for inference
config['N_steps'] = 1000
# Define as [] to loop through all folds, or specify start and end folds i.e. [3, 5] or [5, 5]
config['fold_to_run'] = [1,5]
# The destination directory for saving the pipeline outputs (models, results, plots, etc.)
config['outdir'] = ''
