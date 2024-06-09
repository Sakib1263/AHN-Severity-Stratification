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
