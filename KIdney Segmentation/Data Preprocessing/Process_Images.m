%% Prepare kFold Cross-Validation Data for Kidney AHN
clear;
clc;
SRCdir = 'Data_Subject_Wise_Curated';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
current_fold_num = 1;
IMGoutdir = sprintf('Data_Results_KFold/Data/Train/Fold_%d/Images',current_fold_num);
if ~exist(IMGoutdir, 'dir')
    mkdir(IMGoutdir)
end
MSKoutdir = sprintf('Data_Results_KFold/Data/Train/Fold_%d/Masks',current_fold_num);
if ~exist(MSKoutdir, 'dir')
    mkdir(MSKoutdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
for i=1:length(SRCfoldernames)
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    subject_num = str2double(SRCfolderDir);
    if (subject_num < 1) || (subject_num > 96)
        continue
    end
    SRCindir = fullfile(SRCdir,SRCfolderDir,'Images');
    SRCfiles = dir(SRCindir);
    SRCfilenames = {SRCfiles.name};
    SRCfilenames = SRCfilenames(3:end);
    for ii=1:length(SRCfilenames)
        SRCcurrentinfile = cell2mat(SRCfilenames(ii));
        SRCcurrentinfiledir = fullfile(SRCindir,SRCcurrentinfile);
        CurrentIMG = imread(SRCcurrentinfiledir);
        CurrentIMGSize = size(CurrentIMG);
        if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
            CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
        end
        imwrite(CurrentIMG,IMGoutdir,'BitDepth',8,'Mode','lossless');
    end
    SRCindir = fullfile(SRCdir,SRCfolderDir,'Masks');
    SRCfiles = dir(SRCindir);
    SRCfilenames = {SRCfiles.name};
    SRCfilenames = SRCfilenames(3:end);
    for ii=1:length(SRCfilenames)
        MSKcurrentinfile = cell2mat(SRCfilenames(ii));
        MSKcurrentinfiledir = fullfile(SRCindir,MSKcurrentinfile);
        CurrentMask = imread(MSKcurrentinfiledir);
        CurrentMaskSize = size(CurrentMask);
        if (CurrentMaskSize(1) ~= imheight) || (CurrentMaskSize(2) ~= imwidth)
            CurrentMask = imresize(CurrentMask,[imheight imwidth]);
        end
        CurrentMask = mat2gray(CurrentMask,[0 1]);
        imwrite(CurrentMask,MSKoutdir,'BitDepth',8,'Mode','lossless');
    end
    disp(subject_num);
end
%% Resize Images
clear;
clc;
SRCdir = 'SFU4_Sidra/14/Images';
SRCdinfo = dir(SRCdir);
SRCfilenames = {SRCdinfo.name};
SRCfilenames = SRCfilenames(1:end-2);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
outdir = SRCdir;
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
for i=1:length(SRCfilenames)
    SRCfilename = cell2mat(SRCfilenames(i));
    SRCcurrentinfiledir = fullfile(SRCdir,SRCfilename);
    CurrentIMG = imread(SRCcurrentinfiledir);
    CurrentIMGSize = size(CurrentIMG);
    if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
        CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
    end
    IMGoutdir = fullfile(outdir,SRCfilename);
    imwrite(CurrentIMG,IMGoutdir,'BitDepth',8,'Mode','lossless');
end
%% Prepare kFold Cross-Validation Data for Kidney Segmentation (V2)
clear;
clc;
SRCdir = 'Data_Subject_Wise_Curated_V2';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
current_fold_num = 4;
IMGoutdir = sprintf('Data_Results_KFold/Data/Train/Fold_%d/Images',current_fold_num);
if ~exist(IMGoutdir, 'dir')
    mkdir(IMGoutdir)
end
MSKoutdir = sprintf('Data_Results_KFold/Data/Train/Fold_%d/Masks',current_fold_num);
if ~exist(MSKoutdir, 'dir')
    mkdir(MSKoutdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
for i=1:length(SRCfoldernames)
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    subject_num = str2double(SRCfolderDir);
    if (subject_num > 30) && (subject_num < 85)
        continue
    end
    SRCIMGdir = fullfile(SRCdir,SRCfolderDir,'Images');
    SRCIMGfiles = dir(SRCIMGdir);
    SRCIMGfilenames = {SRCIMGfiles.name};
    SRCIMGfilenames = SRCIMGfilenames(3:end);
    for ii=1:length(SRCIMGfilenames)
        SRCcurrentinfile = cell2mat(SRCIMGfilenames(ii));
        SRCcurrentinfiledir = fullfile(SRCIMGdir,SRCcurrentinfile);
        CurrentIMG = imread(SRCcurrentinfiledir);
        CurrentIMGSize = size(CurrentIMG);
        if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
            CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
        end
        IMGoutput_filedir = fullfile(IMGoutdir,SRCcurrentinfile);
        imwrite(CurrentIMG,IMGoutput_filedir,'BitDepth',8,'Mode','lossless');
    end
    %
    SRCMSKdir = fullfile(SRCdir,SRCfolderDir,'Masks');
    SRCMSKfiles = dir(SRCMSKdir);
    SRCMSKfilenames = {SRCMSKfiles.name};
    SRCMSKfilenames = SRCMSKfilenames(3:end);
    for ii=1:length(SRCMSKfilenames)
        SRCcurrentinfile = cell2mat(SRCMSKfilenames(ii));
        SRCcurrentinfiledir = fullfile(SRCMSKdir,SRCcurrentinfile);
        CurrentMask = imread(SRCcurrentinfiledir);
        CurrentMaskSize = size(CurrentMask);
        if (CurrentMaskSize(1) ~= imheight) || (CurrentMaskSize(2) ~= imwidth)
            CurrentMask = imresize(CurrentMask,[imheight imwidth]);
        end
        if CurrentMask(1,1) > 0
            CurrentMask = imbinarize(CurrentMask);
            CurrentMask = uint8(imcomplement(CurrentMask))*255;
        else
            CurrentMask = imbinarize(CurrentMask);
            CurrentMask = uint8(CurrentMask)*255;
        end
        % SRCcurrentMSKfile = strcat(SRCcurrentinfile(1:end-5),'.png');
        MSKoutput_filedir = fullfile(MSKoutdir,SRCcurrentinfile);
        imwrite(CurrentMask,MSKoutput_filedir,'BitDepth',8,'Mode','lossless');
    end
    disp(subject_num);
end
%% Prepare kFold Cross-Validation Data for AHN Binary Classification
clear;
clc;
SRCdir = 'Data_Subject_Wise_Combined_Updated';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
current_fold_num = 5;
HNoutdir = sprintf('Data_Results_KFold/Classification/Data/Val/Fold_%d/HN',current_fold_num);
if ~exist(HNoutdir, 'dir')
    mkdir(HNoutdir)
end
Normoutdir = sprintf('Data_Results_KFold/Classification/Data/Val/Fold_%d/Normal',current_fold_num);
if ~exist(Normoutdir, 'dir')
    mkdir(Normoutdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
for i=1:length(SRCfoldernames)
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    subject_num = str2double(SRCfolderDir);
    if (subject_num < 31) || (subject_num > 54)
        continue
    end
    SRCIMGdir = fullfile(SRCdir,SRCfolderDir,'HN');
    SRCIMGfiles = dir(SRCIMGdir);
    SRCIMGfilenames = {SRCIMGfiles.name};
    SRCIMGfilenames = SRCIMGfilenames(3:end);
    for ii=1:length(SRCIMGfilenames)
        SRCcurrentinfile = cell2mat(SRCIMGfilenames(ii));
        SRCcurrentinfiledir = fullfile(SRCIMGdir,SRCcurrentinfile);
        CurrentIMG = imread(SRCcurrentinfiledir);
        CurrentIMGSize = size(CurrentIMG);
        if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
            CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
        end
        IMGoutput_filedir = fullfile(HNoutdir,SRCcurrentinfile);
        imwrite(CurrentIMG,IMGoutput_filedir,'BitDepth',8,'Mode','lossless');
    end
    %
    SRCIMGdir = fullfile(SRCdir,SRCfolderDir,'Normal');
    SRCIMGfiles = dir(SRCIMGdir);
    SRCIMGfilenames = {SRCIMGfiles.name};
    SRCIMGfilenames = SRCIMGfilenames(3:end);
    for ii=1:length(SRCIMGfilenames)
        SRCcurrentinfile = cell2mat(SRCIMGfilenames(ii));
        SRCcurrentinfiledir = fullfile(SRCIMGdir,SRCcurrentinfile);
        CurrentIMG = imread(SRCcurrentinfiledir);
        CurrentIMGSize = size(CurrentIMG);
        if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
            CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
        end
        IMGoutput_filedir = fullfile(Normoutdir,SRCcurrentinfile);
        imwrite(CurrentIMG,IMGoutput_filedir,'BitDepth',8,'Mode','lossless');
    end
    disp(subject_num);
end
%% Prepare kFold Cross-Validation Data for AHN Binary Classification
clear;
clc;
SRCdir = 'Data_Subject_Wise_Binary_Classification_Annotated_Full_US';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
current_fold_num = 5;
outdir = 'Data_Results_KFold/Classification/AllData';
if ~exist(outdir, 'dir')
    mkdir(outdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
normal_counter = 0;
HN_counter = 0;
for i=1:length(SRCfoldernames)
    HN_outdir = fullfile(outdir,'HN');
    if ~exist(HN_outdir, 'dir')
        mkdir(HN_outdir)
    end
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    subject_num = str2double(SRCfolderDir);
    SRCIMGdir = fullfile(SRCdir,SRCfolderDir,'HN');
    SRCIMGfiles = dir(SRCIMGdir);
    SRCIMGfilenames = {SRCIMGfiles.name};
    SRCIMGfilenames = SRCIMGfilenames(3:end);
    for ii=1:length(SRCIMGfilenames)
        SRCcurrentinfile = cell2mat(SRCIMGfilenames(ii));
        SRCcurrentinfiledir = fullfile(SRCIMGdir,SRCcurrentinfile);
        CurrentIMG = imread(SRCcurrentinfiledir);
        CurrentIMGSize = size(CurrentIMG);
        if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
            CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
        end
        infilename = string(i) + '_' + string(ii) + '.png';
        IMGoutput_filedir = fullfile(HN_outdir,infilename);
        imwrite(CurrentIMG,IMGoutput_filedir,'BitDepth',8,'Mode','lossless');
    end
    %
    normal_outdir = fullfile(outdir,'Normal');
    if ~exist(normal_outdir, 'dir')
        mkdir(normal_outdir)
    end
    SRCIMGdir = fullfile(SRCdir,SRCfolderDir,'Normal');
    SRCIMGfiles = dir(SRCIMGdir);
    SRCIMGfilenames = {SRCIMGfiles.name};
    SRCIMGfilenames = SRCIMGfilenames(3:end);
    for ii=1:length(SRCIMGfilenames)
        SRCcurrentinfile = cell2mat(SRCIMGfilenames(ii));
        SRCcurrentinfiledir = fullfile(SRCIMGdir,SRCcurrentinfile);
        CurrentIMG = imread(SRCcurrentinfiledir);
        CurrentIMGSize = size(CurrentIMG);
        if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
            CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
        end
        infilename = string(i) + '_' + string(ii) + '.png';
        IMGoutput_filedir = fullfile(normal_outdir,infilename);
        imwrite(CurrentIMG,IMGoutput_filedir,'BitDepth',8,'Mode','lossless');
    end
    disp(subject_num);
end
%% Prepare Whole Dataset with Image-Mask Separate Folder per Subject
clear;
clc;
SRCdir = 'Data_Subject_Wise_Curated';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
OUTdir = 'Data_Subject_Wise_Curated_V2';
if ~exist(OUTdir, 'dir')
    mkdir(OUTdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
for i=1:length(SRCfoldernames)
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    subject_num = str2double(SRCfolderDir);
    SRCindir = fullfile(SRCdir,SRCfolderDir);
    SRCfiles = dir(SRCindir);
    SRCfilenames = {SRCfiles.name};
    SRCfilenames = SRCfilenames(3:end);
    IMGOUTdir = fullfile(OUTdir,SRCfolderDir,'Images');
    if ~exist(IMGOUTdir, 'dir')
        mkdir(IMGOUTdir)
    end
    MSKOUTdir = fullfile(OUTdir,SRCfolderDir,'Masks');
    if ~exist(MSKOUTdir, 'dir')
        mkdir(MSKOUTdir)
    end
    for ii=1:length(SRCfilenames)
        SRCcurrentinfile = cell2mat(SRCfilenames(ii));
        SRCcurrentinfileABBR = SRCcurrentinfile(end-2:end);
        IMGorMSK = SRCcurrentinfile(end-4:end-4);
        SRCcurrentinfiledir = fullfile(SRCindir,SRCcurrentinfile);
        if SRCcurrentinfileABBR ~= "png"
            fprintf('There is a non-PNG file in this location: %s',SRCcurrentinfiledir);
            return
        end
        if (IMGorMSK == 'a') || (IMGorMSK == 'A')
            MSKcurrentinfile = cell2mat(SRCfilenames(ii));
            MSKcurrentinfiledir = fullfile(SRCindir,MSKcurrentinfile);
            CurrentMask = imread(MSKcurrentinfiledir);
            CurrentMaskSize = size(CurrentMask);
            if (CurrentMaskSize(1) ~= imheight) || (CurrentMaskSize(2) ~= imwidth)
                CurrentMask = imresize(CurrentMask,[imheight imwidth]);
            end
            if CurrentMask(1,1) > 0
                CurrentMask = imbinarize(CurrentMask);
                CurrentMask = uint8(imcomplement(CurrentMask))*255;
            else
                CurrentMask = imbinarize(CurrentMask);
                CurrentMask = uint8(CurrentMask)*255;
            end
            SRCcurrentMSKfile = strcat(SRCcurrentinfile(1:end-5),'.png');
            MSKoutput_filedir = fullfile(MSKOUTdir,SRCcurrentMSKfile);
            imwrite(CurrentMask,MSKoutput_filedir,'BitDepth',8,'Mode','lossless');
        else
            CurrentIMG = imread(SRCcurrentinfiledir);
            CurrentIMGSize = size(CurrentIMG);
            if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
                CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
            end
            IMGoutput_filedir = fullfile(IMGOUTdir,SRCcurrentinfile);
            imwrite(CurrentIMG,IMGoutput_filedir,'BitDepth',8,'Mode','lossless');
        end
    end
    disp(subject_num);
end
%% Prepare Whole Dataset with Image-Mask Separate Folder per Subject
clear;
clc;
SRCdir = 'Data_Subject_Wise_Curated';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
OUTdir = 'Data_Subject_Wise_Curated_V2';
if ~exist(OUTdir, 'dir')
    mkdir(OUTdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
for i=1:length(SRCfoldernames)
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    subject_num = str2double(SRCfolderDir);
    SRCindir = fullfile(SRCdir,SRCfolderDir);
    SRCfiles = dir(SRCindir);
    SRCfilenames = {SRCfiles.name};
    SRCfilenames = SRCfilenames(3:end);
    IMGOUTdir = fullfile(OUTdir,SRCfolderDir,'Images');
    if ~exist(IMGOUTdir, 'dir')
        mkdir(IMGOUTdir)
    end
    MSKOUTdir = fullfile(OUTdir,SRCfolderDir,'Masks');
    if ~exist(MSKOUTdir, 'dir')
        mkdir(MSKOUTdir)
    end
    for ii=1:length(SRCfilenames)
        SRCcurrentinfile = cell2mat(SRCfilenames(ii));
        SRCcurrentinfileABBR = SRCcurrentinfile(end-2:end);
        IMGorMSK = SRCcurrentinfile(end-4:end-4);
        SRCcurrentinfiledir = fullfile(SRCindir,SRCcurrentinfile);
        if SRCcurrentinfileABBR ~= "png"
            fprintf('There is a non-PNG file in this location: %s',SRCcurrentinfiledir);
            return
        end
        if (IMGorMSK == 'a') || (IMGorMSK == 'A')
            MSKcurrentinfile = cell2mat(SRCfilenames(ii));
            MSKcurrentinfiledir = fullfile(SRCindir,MSKcurrentinfile);
            CurrentMask = imread(MSKcurrentinfiledir);
            CurrentMaskSize = size(CurrentMask);
            if (CurrentMaskSize(1) ~= imheight) || (CurrentMaskSize(2) ~= imwidth)
                CurrentMask = imresize(CurrentMask,[imheight imwidth]);
            end
            if CurrentMask(1,1) > 0
                CurrentMask = imbinarize(CurrentMask);
                CurrentMask = uint8(imcomplement(CurrentMask))*255;
            else
                CurrentMask = imbinarize(CurrentMask);
                CurrentMask = uint8(CurrentMask)*255;
            end
            SRCcurrentMSKfile = strcat(SRCcurrentinfile(1:end-5),'.png');
            MSKoutput_filedir = fullfile(MSKOUTdir,SRCcurrentMSKfile);
            imwrite(CurrentMask,MSKoutput_filedir,'BitDepth',8,'Mode','lossless');
        else
            CurrentIMG = imread(SRCcurrentinfiledir);
            CurrentIMGSize = size(CurrentIMG);
            if (CurrentIMGSize(1) ~= imheight) || (CurrentIMGSize(2) ~= imwidth)
                CurrentIMG = imresize(CurrentIMG,[imheight imwidth]);
            end
            IMGoutput_filedir = fullfile(IMGOUTdir,SRCcurrentinfile);
            imwrite(CurrentIMG,IMGoutput_filedir,'BitDepth',8,'Mode','lossless');
        end
    end
    disp(subject_num);
end
%% Combine Kidney Images and Masks
clear;
clc;
SRCdir = 'Data_Subject_Wise_Curated_V2';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
OUTdir = 'Data_Subject_Wise_Kidney_only';
if ~exist(OUTdir, 'dir')
    mkdir(OUTdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
for i=1:length(SRCfoldernames)
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    subject_num = str2double(SRCfolderDir);
    SRCIMGdir = fullfile(SRCdir,SRCfolderDir,'Images');
    SRCIMGfiles = dir(SRCIMGdir);
    SRCIMGfiles = {SRCIMGfiles.name};
    SRCIMGfilenames = sort(SRCIMGfiles(3:end));
    SRCMSKdir = fullfile(SRCdir,SRCfolderDir,'Masks');
    SRCMSKfiles = dir(SRCMSKdir);
    SRCMSKfiles = {SRCMSKfiles.name};
    SRCMSKfilenames = sort(SRCMSKfiles(3:end));
    for ii=1:length(SRCIMGfilenames)
        SRCcurrentIMGfile = cell2mat(SRCIMGfilenames(ii));
        SRCcurrentIMGdir = fullfile(SRCIMGdir,SRCcurrentIMGfile);
        SRCcurrentMSKfile = cell2mat(SRCMSKfilenames(ii));
        SRCcurrentMSKdir = fullfile(SRCMSKdir,SRCcurrentMSKfile);
        CurrentImage = imread(SRCcurrentIMGdir);
        CurrentMask = imread(SRCcurrentMSKdir);
        % Loop on RGB channels
        for iii=1:3
            CurrentChannel = CurrentImage(:,:,iii);
            CurrentChannel(~CurrentMask) = 0;
            if iii == 1
                MaskedRGBImage = CurrentChannel;
            else
                MaskedRGBImage = cat(3,MaskedRGBImage,CurrentChannel);
            end
        end
        IMGOUTdir = fullfile(OUTdir,SRCfolderDir);
        if ~exist(IMGOUTdir, 'dir')
            mkdir(IMGOUTdir)
        end
        IMGfiledir = fullfile(IMGOUTdir,SRCcurrentIMGfile);
        imwrite(MaskedRGBImage,IMGfiledir,'BitDepth',8,'Mode','lossless');
    end
    disp(subject_num);
end
%% Combine Kidney Images and Masks v2
clear;
clc;
SRCdir = 'SFU4_Sidra_Annotated';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
OUTdir = 'SFU4_Sidra_Combined';
if ~exist(OUTdir, 'dir')
    mkdir(OUTdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
for i=1:length(SRCfoldernames)
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    subject_num = str2double(SRCfolderDir);
    SRCIMGdir = fullfile(SRCdir,SRCfolderDir,'Images');
    SRCIMGfiles = dir(SRCIMGdir);
    SRCIMGfiles = {SRCIMGfiles.name};
    SRCIMGfilenames = sort(SRCIMGfiles(3:end));
    SRCMSKdir = fullfile(SRCdir,SRCfolderDir,'Masks');
    SRCMSKfiles = dir(SRCMSKdir);
    SRCMSKfiles = {SRCMSKfiles.name};
    SRCMSKfilenames = sort(SRCMSKfiles(3:end));
    for ii=1:length(SRCIMGfilenames)
        SRCcurrentIMGfile = cell2mat(SRCIMGfilenames(ii));
        SRCcurrentIMGdir = fullfile(SRCIMGdir,SRCcurrentIMGfile);
        SRCcurrentMSKfile = cell2mat(SRCMSKfilenames(ii));
        SRCcurrentMSKdir = fullfile(SRCMSKdir,SRCcurrentMSKfile);
        CurrentImage = imread(SRCcurrentIMGdir);
        CurrentMask = im2gray(imread(SRCcurrentMSKdir));
        % Loop on RGB channels
        for iii=1:3
            CurrentChannel = CurrentImage(:,:,iii);
            CurrentChannel(~CurrentMask) = 0;
            if iii == 1
                MaskedRGBImage = CurrentChannel;
            else
                MaskedRGBImage = cat(3,MaskedRGBImage,CurrentChannel);
            end
        end
        IMGOUTdir = fullfile(OUTdir,SRCfolderDir);
        if ~exist(IMGOUTdir, 'dir')
            mkdir(IMGOUTdir)
        end
        IMGfiledir = fullfile(IMGOUTdir,SRCcurrentIMGfile);
        imwrite(MaskedRGBImage,IMGfiledir,'BitDepth',8,'Mode','lossless');
    end
    disp(subject_num);
end
%% Create Dataset for Binary Classification
clear;
clc;
SRCdir = 'Data_Subject_Wise_Curated_V2';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
annotation_dir = 'Data_Subject_Wise_Binary_Classification_Annotated_Kidney_Only';
annotationdinfo = dir(annotation_dir);
annotationfoldernames = {annotationdinfo.name};
annotationfoldernames = annotationfoldernames(3:end);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
OUTdir = 'Data_Subject_Wise_Binary_Classification_Annotated_Full_US';
if ~exist(OUTdir, 'dir')
    mkdir(OUTdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
total_counter = 0;
normal_counter = 0;
HN_counter = 0;
for i=1:length(SRCfoldernames)
    disp(i);
    SRCfolderDir = cell2mat(SRCfoldernames(i));
    current_src_subject_fulldir = fullfile(SRCdir,SRCfolderDir,'Images');
    current_src_subject_info = dir(current_src_subject_fulldir);
    current_src_image_names = {current_src_subject_info.name};
    current_src_image_names = current_src_image_names(3:end);
    %
    current_annotation_subject = cell2mat(annotationfoldernames(i));
    current_annotation_subject_fulldir = fullfile(annotation_dir,current_annotation_subject);
    current_annotation_subject_info = dir(current_annotation_subject_fulldir);
    current_annotation_subfolder_names = {current_annotation_subject_info.name};
    current_annotation_subfolder_names = current_annotation_subfolder_names(3:end);
    for ii=1:length(current_annotation_subfolder_names)
        current_annotation_subject = cell2mat(annotationfoldernames(i));
        current_annotation_subject_fulldir = fullfile(annotation_dir,current_annotation_subject,cell2mat(current_annotation_subfolder_names(ii)));
        us_image_info = dir(current_annotation_subject_fulldir);
        us_image_info_names = {us_image_info.name};
        us_image_info_names = us_image_info_names(3:end);
        for iii=1:length(us_image_info_names)
            us_image_fulldir = fullfile(current_annotation_subject_fulldir,cell2mat(us_image_info_names(iii)));
            us_current_image_name = cell2mat(us_image_info_names(iii));
            %disp(us_current_image_name);
            for iv=1:length(current_src_image_names)
                current_src_image_name = cell2mat(current_src_image_names(iv));
                subjects_binary_class = cell2mat(current_annotation_subfolder_names(ii));
                src_us_image_dir = fullfile(current_src_subject_fulldir,current_src_image_name);
                save_dir = fullfile(OUTdir,current_annotation_subject,subjects_binary_class);
                if ~exist(save_dir, 'dir')
                    mkdir(save_dir)
                end
                if strcmp(us_current_image_name,current_src_image_name) == 1
                    total_counter = total_counter + 1;
                    if subjects_binary_class == "Normal"
                        normal_counter = normal_counter + 1;
                        copyfile(src_us_image_dir,save_dir);
                    elseif subjects_binary_class == "HN"
                        HN_counter = HN_counter + 1;
                        copyfile(src_us_image_dir,save_dir);
                    end
                end
            end
        end
    end
end
disp(total_counter);
disp(normal_counter);
disp(HN_counter);
%% Create Dataset for Binary Classification V2
clear;
clc;
SRCdir = 'Data_Subject_Wise_Binary_Classification_Annotated_Full_US';
SRCdinfo = dir(SRCdir);
SRCfoldernames = {SRCdinfo.name};
SRCfoldernames = SRCfoldernames(3:end);
imheight = 512; % Original Height = 512
imwidth = 512; % Original Width = 512
OUTdir = 'Data_Results_KFold/Classification/AllData/Full_US_Images_Mathced';
if ~exist(OUTdir, 'dir')
    mkdir(OUTdir)
end
% Loop through all Folders
% i=1:length(IMGfoldernames) % Assuming same folder names are present in both image and mask directories, as it is here
total_counter = 0;
normal_counter = 0;
HN_counter = 0;
for i=1:length(SRCfoldernames)
    disp(i);
    current_src_subject = cell2mat(SRCfoldernames(i));
    current_src_subject_fulldir = fullfile(SRCdir,current_src_subject);
    current_src_subject_info = dir(current_src_subject_fulldir);
    current_annotation_subfolder_names = {current_src_subject_info.name};
    current_annotation_subfolder_names = current_annotation_subfolder_names(3:end);
    for ii=1:length(current_annotation_subfolder_names)
        current_annotation_subject = cell2mat(SRCfoldernames(i));
        current_annotation_subject_fulldir = fullfile(SRCdir,current_annotation_subject,cell2mat(current_annotation_subfolder_names(ii)));
        current_src_image_info = dir(current_annotation_subject_fulldir);
        current_src_image_names = {current_src_image_info.name};
        current_src_image_names = current_src_image_names(3:end);
        for iii=1:length(current_src_image_names)
            us_image_fulldir = fullfile(current_annotation_subject_fulldir,cell2mat(current_src_image_names(iii)));
            current_src_image_name = cell2mat(current_src_image_names(iii));
            %disp(us_current_image_name);
            subjects_binary_class = cell2mat(current_annotation_subfolder_names(ii));
            src_us_image_dir = fullfile(current_annotation_subject_fulldir,current_src_image_name);
            save_dir = fullfile(OUTdir,subjects_binary_class);
            if ~exist(save_dir, 'dir')
                mkdir(save_dir)
            end
            total_counter = total_counter + 1;
            if subjects_binary_class == "Normal"
                normal_counter = normal_counter + 1;
                IMGfiledir = fullfile(save_dir,strcat(current_src_subject,'_',string(iii),'.png'));
                current_image = imread(src_us_image_dir);
                imwrite(current_image,IMGfiledir,'BitDepth',8,'Mode','lossless');
            elseif subjects_binary_class == "HN"
                HN_counter = HN_counter + 1;
                IMGfiledir = fullfile(save_dir,strcat(current_src_subject,'_',string(iii),'.png'));
                current_image = imread(src_us_image_dir);
                imwrite(current_image,IMGfiledir,'BitDepth',8,'Mode','lossless');
            end
        end
    end
end
disp(total_counter);
disp(normal_counter);
disp(HN_counter);
%% Check Image Similarity and Match
clear;
clc;
SRCdir1 = 'Data_Results_KFold/Classification/AllData/Full_US_Images/HN';
SRCdir1info = dir(SRCdir1);
SRC1filenames = {SRCdir1info.name};
SRC1filenames = SRC1filenames(3:end);
SRCdir2 = 'Data_Results_KFold/Classification/AllData/Full_US_Images_Mathced/HN';
SRCdir2info = dir(SRCdir2);
SRC2filenames = {SRCdir2info.name};
SRC2filenames = SRC2filenames(3:end);
SRCdir3 = 'Data_Results_KFold/Classification/AllData/Combined_US_Images/HN';
SRCdir3info = dir(SRCdir3);
SRC3filenames = {SRCdir3info.name};
SRC3filenames = SRC3filenames(3:end);
outdir = 'Data_Results_KFold/Classification/AllData/Combined_US_Images_Matched/HN';
if ~exist(outdir, 'dir')
    mkdir(outdir)
end
for i=1:length(SRC1filenames)
    template_img_name = cell2mat(SRC1filenames(i));
    template_img_full_dir = fullfile(SRCdir1,template_img_name);
    template_img = im2gray(imread(template_img_full_dir));
    img_max_error = inf;
    for ii=1:length(SRC2filenames)
        target_img_name = cell2mat(SRC2filenames(ii));
        target_img_full_dir = fullfile(SRCdir2,target_img_name);
        target_img = im2gray(imread(target_img_full_dir));
        current_error = immse(template_img,target_img);
        if current_error < img_max_error
            img_max_error = current_error;
            matched_img = target_img_name;
        end
    end
    fprintf('The new img %s matches most with the original img %s with an MSE of %f\n',matched_img,template_img_name,img_max_error);
    for ii=1:length(SRC3filenames)
        combined_img_name = cell2mat(SRC3filenames(ii));
        if strcmp(combined_img_name,matched_img) == 1
            combined_us_image_indir = fullfile(SRCdir3,combined_img_name);
            current_combined_image = imread(combined_us_image_indir);
            combined_us_image_svdir = fullfile(outdir,template_img_name);
            imwrite(current_combined_image,combined_us_image_svdir,'BitDepth',8,'Mode','lossless');
        end
    end
end
%% Mask Inversion
clear;
clc;
root_dir = 'White_Mask'; 
out_dir = 'Black_Mask';
if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end
image_file_list = dir(root_dir);
image_file_names = {image_file_list.name};
image_file_names = image_file_names(3:end);
% i=1:length(image_file_names)
for i=1:length(image_file_names)
    image_file_dir = cell2mat(fullfile(root_dir,image_file_names(i)));
    gt_mask = im2gray(imread(image_file_dir));
    gt_mask1 = imbinarize(gt_mask,0.5);
    if gt_mask1(1,1) == 1
        inverted_mask = uint8(imcomplement(gt_mask1))*255;
    else
        inverted_mask = uint8(gt_mask1)*255;
    end
    output_filename = cell2mat(image_file_names(i));
    output_filedir = fullfile(out_dir,output_filename);
    imwrite(inverted_mask,output_filedir,'BitDepth',8,'Mode','lossless');
end
%% Return file list and save as a CSV
clear;
clc;
SRCdir = 'Data_Results_KFold/Classification/AllData/HN';
OUTdir = 'Data_Results_KFold/Classification/AllData_Shuffled_New/HN';
if ~exist(OUTdir, 'dir')
    mkdir(OUTdir)
end
SRCdinfo = dir(SRCdir);
SRCfilenames = {SRCdinfo.name};
SRCfilenames = SRCfilenames(3:end);
% writecell(SRCfilenames,'Filenames.csv')
SRCfilenames = SRCfilenames(randperm(length(SRCfilenames)));
numSRCfiles = length(SRCfilenames);
for i=1:numSRCfiles
    SRCfilename_temp = cell2mat(SRCfilenames(i));
    SRCfilename_temp_fulldir = fullfile(SRCdir,SRCfilename_temp);
    SRCfilename_temp_new = string(i) + '_' + SRCfilename_temp;
    OUTdir_full = fullfile(OUTdir,SRCfilename_temp_new);
    copyfile(SRCfilename_temp_fulldir,OUTdir_full);
    disp(SRCfilename_temp_fulldir);
end
%%
clear;
clc;
SRCdir = 'Data_Results_KFold/Classification/AllData_Shuffled/HN';
SRCdinfo = dir(SRCdir);
SRCfilenames = {SRCdinfo.name};
SRCfilenames = SRCfilenames(3:end);
numSRCfiles = length(SRCfilenames);
all_subnames = zeros(numSRCfiles,1);
for i=1:numSRCfiles
    SRCfilename_temp = cell2mat(SRCfilenames(i));
    SRCfilename_temps = strsplit(SRCfilename_temp,'_');
    current_subname = str2num(cell2mat(SRCfilename_temps(2)));
    all_subnames(i,1) = current_subname;
end
writematrix(all_subnames,'All_Subnames_ShuffledData.xlsx');
disp(all_subnames);
%% the loaded data
clear;
clc;
% image and masks path
image_path = 'Data_Results_Phase_1\Classification\AllData\Combined_US_Images\HN';
image_file=dir(fullfile(image_path,'\*'));
image_file = struct2cell(image_file);
images_name= image_file(1,3:end); 

gray_image = true; % set to true if you have gray scale images 

% resize dimensions 
resize_images = true;    % set to true to resize images 
Y_size = 512;
X_size = 512; 


m = length(images_name);
ch_1_mean = 0.0;
ch_2_mean = 0.0;
ch_3_mean = 0.0;
ch_1_std = 0.0;
ch_2_std = 0.0;
ch_3_std = 0.0;
for k=1:m
    
    % read image
    F_image = fullfile(image_path,images_name{k});
    I_image = imread(F_image);
    % resize 
    if resize_images==1
        I_image = imresize(I_image, [Y_size X_size]);
    end
    % rgb to gray
    if gray_image==1
        if ndims(I_image)==3
            I_image = rgb2gray(I_image);
        end
        I_image = cat(3,I_image,I_image,I_image);
    end
    
    ch_1 = double(I_image(:,:,1))/255.0;
    ch_1_mean = ch_1_mean + mean(ch_1(:));
    ch_1_std = ch_1_std + std(ch_1(:));
    
    ch_2 = double(I_image(:,:,2))/255.0;
    ch_2_mean = ch_2_mean + mean(ch_2(:));
    ch_2_std = ch_2_std + std(ch_2(:));
    
    ch_3 = double(I_image(:,:,3))/255.0;
    ch_3_mean = ch_3_mean + mean(ch_3(:));
    ch_3_std = ch_3_std + std(ch_3(:));
    
end

ch_1_mean = ch_1_mean/m;
ch_2_mean = ch_2_mean/m;
ch_3_mean = ch_3_mean/m;

ch_1_std = ch_1_std/m;
ch_2_std = ch_2_std/m;
ch_3_std = ch_3_std/m;

disp('mean');
disp([ch_1_mean, ch_2_mean, ch_3_mean]);
disp('std');
disp([ch_1_std, ch_2_std, ch_3_std]);