%% the loaded data
clear;
% image and masks path
image_path = 'Data/Train/Fold_1/NoKidney';
image_file = dir(fullfile(image_path,'\*'));
image_file = struct2cell(image_file);
images_name= image_file(1,3:end); 
gray_image = false; % set to true if you have gray scale images 
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