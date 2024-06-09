%% Rename random files to .dcm or .dicom in the end to convert to PNG/JPG,etc.
clear;
clc;
srcprojectdir = 'RawData_Annonymous';
svprojectdir = 'RawData_Annonymous_DCM';
dinfo = dir(srcprojectdir);
foldernames = {dinfo.name};
foldernames = foldernames(3:end);
% Loop through all Folders
for i=1:length(foldernames)
    foldername = cell2mat(foldernames(i));
    src_subjectdir = fullfile(srcprojectdir,foldername);
	dinfo = dir(src_subjectdir);
    oldfilenames = {dinfo.name};
    oldfilenames = oldfilenames(3:end);
    sv_subjectdir = fullfile(svprojectdir,foldername);
    if ~exist(sv_subjectdir,'dir')
        mkdir(sv_subjectdir)
    end
    for ii=1:length(oldfilenames)
        oldfilename = cell2mat(oldfilenames(ii));
        % newfilename = oldfilename(1:end-6);  % Remove suffixes from previous filetype
        newfilename = strcat(string(foldername),'_',string(ii),'.dcm');
        movefile(fullfile(src_subjectdir,oldfilename),fullfile(sv_subjectdir,newfilename) );
    end
end
%% Convert DICOM files to PNG/JPG format
clear;
clc;
DICOMdir = 'RawData_Annonymous';
dinfo = dir(DICOMdir);
foldernames = {dinfo.name};
foldernames = foldernames(3:end);
OutputImagedir = 'CuratedDataPNG';
if ~exist(OutputImagedir, 'dir')
    mkdir(OutputImagedir)
end
% Loop through all Folders
for i=1:length(foldernames)
    disp(i);
    foldername = cell2mat(foldernames(i));
    indir = fullfile(DICOMdir,foldername);  % Input folder name with DICOM files.
    outdir = fullfile(OutputImagedir,foldername); % Output folder name for storing jpg files.
    if ~exist(outdir,'dir')
        mkdir(outdir)
    end
    dinfo = dir(indir);
    filenames = {dinfo.name};
    filenames = filenames(3:end);
    for ii=1:length(filenames)
        input_filename = filenames(ii);
        dicom_input_filedir = cell2mat(fullfile(indir,input_filename));
        output_filename = cell2mat(filenames(ii));
        output_filename = output_filename(1:end-4);  % Remove .dcm suffix
        output_filename = [output_filename,'.png'];  % Include .png in filename
        dicom_output_filedir = fullfile(outdir,output_filename);
        image = dicomread(dicom_input_filedir);
        try
            imwrite(image,dicom_output_filedir,'BitDepth',16,'Mode','lossless')
        catch
            continue;
        end
    end
end
%% Combine Subjectwise Dataset for Object Detection (YOLO) Framework
clear;
clc;
SRCdir = 'CuratedDataPNG';
dinfo = dir(SRCdir);
foldernames = {dinfo.name};
foldernames = foldernames(3:end);
OutputImagedir = 'CuratedDataYOLO';
if ~exist(OutputImagedir, 'dir')
    mkdir(OutputImagedir)
end
% Loop through all Folders
for i=1:length(foldernames)
    disp(i);
    foldername = cell2mat(foldernames(i));
    indir = fullfile(SRCdir,foldername);  % Input folder name with DICOM files.
    outdir = fullfile(OutputImagedir,'Images'); % Output folder name for storing jpg files.
    if ~exist(outdir,'dir')
        mkdir(outdir)
    end
    dinfo = dir(indir);
    filenames = {dinfo.name};
    filenames = filenames(3:end);
    for ii=1:length(filenames)
        input_filename = filenames(ii);
        png_input_filedir = cell2mat(fullfile(indir,input_filename));
        output_filename = cell2mat(filenames(ii));
        output_filename = output_filename(1:end-4);  % Remove .dcm suffix
        output_filename = [output_filename,'.png'];  % Include .png in filename
        png_output_filedir = fullfile(outdir,output_filename);
        image = imread(png_input_filedir);
        try
            imwrite(image,png_output_filedir,'BitDepth',16,'Mode','lossless')
        catch
            continue;
        end
    end
end
%% Resize All Images
clear;
clc;
ORGdir = 'CuratedDataPNG';
dinfo = dir(ORGdir);
foldernames = {dinfo.name};
foldernames = foldernames(3:end);
OutputImagedir = 'CuratedDataPNG_Resized';
if ~exist(OutputImagedir, 'dir')
    mkdir(OutputImagedir)
end
% Loop through all Folders
for i=1:length(foldernames)
    foldername = cell2mat(foldernames(i));
    indir = fullfile(ORGdir,foldername);  %Input folder name with DICOM files.
    outdir = fullfile(OutputImagedir,foldername); %Output folder name for storing jpg files.
    if ~exist(outdir, 'dir')
        mkdir(outdir)
    end
    dinfo = dir(indir);
    filenames = {dinfo.name};
    filenames = filenames(3:end);
    for ii=1:length(filenames)
        input_filename = filenames(ii);
        input_filedir = cell2mat(fullfile(indir,input_filename));
        output_filename = cell2mat(filenames(ii));
        output_filedir = fullfile(outdir,output_filename);
        input_image = imread(input_filedir);
        resized_image = imresize(input_image,[512 512]);
        imwrite(resized_image,output_filedir,'BitDepth',16,'Mode','lossless')
    end
end
%% Return HC Numbers
%% Convert DICOM files to PNG/JPG format
clear;
clc;
dir_sub = 'CuratedDataPNG_Resized_Renamed';
dinfo = dir(dir_sub);
foldernames = {dinfo.name};
foldernames = foldernames(3:end)';
filename = 'HMC_HC_Numbers.xlsx';
writecell(foldernames,filename,'Sheet',1)