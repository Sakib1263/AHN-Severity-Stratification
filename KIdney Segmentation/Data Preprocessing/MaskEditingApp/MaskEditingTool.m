clc;
clear;
close all;
%
N_points = 100;  % number of adjustable beads on the mask
mask_path = 'Old Masks\';
image_path = 'Images\';
newmask_path = 'New Masks';
Reject_path = 'Rejected Images'; 
Unsure_path = 'Unsure Images';
first_itr = false;  % set to true if you are modifiing cnn output, else set to false if you are adjusting pre-modified masks
image_size = 512;
resolution_factor = image_size/150.0;  % 150 pixels are used for each inch by default
% create the folders to save new images:
if ~exist(newmask_path, 'dir')
    mkdir(newmask_path)
end
if ~exist(Reject_path, 'dir')
    mkdir(Reject_path)
end
if ~exist(Unsure_path, 'dir')
    mkdir(Unsure_path)
end
%
data_image = dir(image_path);
data_mask = dir(mask_path);
data_image = natsortfiles(data_image(3:end));
data_mask = natsortfiles(data_mask(3:end));
mm = length(data_mask);
% load page index if continuation
fprintf('You have total of %d images \n',mm);
if isfile('PageNumber.mat')
    load('PageNumber.mat','i');
    fprintf('You stopped in image %d \n',i);
    x = input('Would you like to continue from where you stopped? [y/n]: ','s');
    if x=='y'
        x = i+1;
    else
        x = input('Enter the image no. you want to start with: ');
    end
else
    x = input('Enter the image no. you want to start with: ');
end
%
for i=x:length(data_mask)
    fprintf("Image ID: %s \n", data_image(i).name)
    fprintf("Mask ID: %s \n", data_mask(i).name)
    clear I M A1 x1 index;
    % Read image and mask
    if exist(fullfile(newmask_path, data_mask(i).name), 'file')
        message = sprintf('You modified this image previously. \n Would you like to modify your last adjustment?');
        answer = MFquestdlg([0.4 0.7], message, 'Redo', 'Yes (modify)', 'No (redo)', 'Yes (modify)');
        if strcmp(answer, 'Yes (modify)')
            M = imread(fullfile(newmask_path, data_mask(i).name));  % Reading Mask
        elseif strcmp(answer, 'No (redo)')
            M = imread(fullfile(mask_path, data_mask(i).name));  % Reading Mask 
        else
            break;
        end
    else
        M = imread(fullfile(mask_path, data_mask(i).name));  % Reading Mask 
    end
    M = imresize(M, [image_size,image_size]);
    temp_M = M;
    temp_M(temp_M > 0) = 255;
    I = imread(fullfile(image_path, data_image(i).name));
    I = imresize(I, [image_size,image_size]);
    if ndims(I) == 3
        I = rgb2gray(I);
    end
    % find mask boundries
    BW = imbinarize(double(temp_M));
    BW_filled = im2gray(double(imfill(BW,'holes')));
    boundaries = bwboundaries(BW_filled);
    if ~isempty(boundaries)
        A = [];
        for k=1:length(boundaries)
            b = boundaries{k};
            A = [A; length(boundaries{k})];
        end
        A1 = max(A);
        u=0;
        for s=1:length(A)
            if(A(s)==A1)
                u=u+1;
            end
        end 
        x1 = find(A==A1);
        A1 = boundaries{x1};
        if length(A1) < 3
            A1=[1 1; 5 5; 10 10; 15 15; 20 20];
        end
    else
        A1=[1 1; 5 5; 10 10; 15 15; 20 20];
    end
    filename = 'XY_positions.mat';
    save(filename,'A1');
    if length(A1) > N_points  % Check if code could not find boundaries
        step = length(A1)/N_points;
        step = ceil(step);
        %
        index = 1;
        j=1;
        while index(end) < length(A1)
            index = [index; j*step];
            j = j + 1;
        end
        if index(end) > length(A1)
            index(end) = length(A1);
        end
        ROI1_temp = A1(index,:);
        ROI1 = cat(2,ROI1_temp(:,2),ROI1_temp(:,1));
        %
        save('ROI1.mat','ROI1');
    else
        ROI1_temp=A1;
        clear ROI1;
        ROI1 = cat(2,ROI1_temp(:,2),ROI1_temp(:,1));
        save('ROI1.mat','ROI1');
    end
    % display image
    if first_itr
        % plot
        close all;
        figure;
        set(gcf, 'Position', get(0, 'Screensize'));
        tit = strcat(data_mask(i).name(1:end-4),'...[',num2str(i),'/',num2str(mm),']');
        sgtitle(tit);
        % subplot 1
        subplot(1,2,1)
        imshow(M);
        hold on;
        plot(ROI1(:,1),ROI1(:,2),'g');
        hold off;
        % subplot 2
        subplot(1,2,2)
        imshow(I);
        hold on;
        plot(ROI1(:,1),ROI1(:,2),'g');
        hold off;
    else
        I_mask = cat(3, I, I, I);
        for jj=1:size(A1,1)
            I_mask(A1(jj,1), A1(jj,2), :)=[0 255 0];
        end
        % plot
        close all;
        figure;
        set(gcf, 'Position', get(0, 'Screensize'));
        tit = strcat(data_mask(i).name(1:end-4),'...[',num2str(i),'/',num2str(mm),']');
        sgtitle(tit);
        % subplot 1
        subplot(1,2,1)
        imshow(M);
        % subplot 2
        subplot(1,2,2)
        imshow(I_mask);
    end
    message = sprintf('Would you like to Continue, Edit or Reject?');
    answer = MFquestdlg([0.4 0.7],message,'MaskEdit', 'Continue', 'Edit', 'Reject', 'Continue');
    if strcmp(answer, 'Continue')
        if first_itr
            figure;
            set(gcf,'PaperUnits','inches','PaperPosition',[0 0 resolution_factor resolution_factor]);
            axes('Units', 'normalized', 'Position', [0 0 1 1]);
            p = zeros(image_size,image_size);
            imshow(p);
            hold on;
            patch = fill(ROI1(:,1),ROI1(:,2),'w');
            patch.EdgeColor = "w";
            patch.LineWidth = 2;
            hold off;
            saveas(gcf, strcat(newmask_path, '\', data_mask(i).name),'png');
            if exist(fullfile(Reject_path, data_mask(i).name), 'file')
                delete(fullfile(Reject_path, data_mask(i).name));
            end
            if exist(fullfile(Unsure_path, data_mask(i).name), 'file')
                delete(fullfile(Unsure_path, data_mask(i).name));
            end
            close all;
        else
            imwrite(M, strcat(newmask_path, '\', data_mask(i).name));
            if exist(fullfile(Reject_path, data_mask(i).name), 'file')
                delete(fullfile(Reject_path, data_mask(i).name));
            end
            if exist(fullfile(Unsure_path, data_mask(i).name), 'file')
                delete(fullfile(Unsure_path, data_mask(i).name));
            end
        end
    elseif strcmp(answer, 'Edit')
        answer='No';
        while(strcmp(answer,'No'))
            close all;
            figure;
            set(gcf, 'Position', get(0, 'Screensize'));
            % subplot 1
            subplot(1,2,1);
            imshow(I)
            sgtitle(tit);
            % subplot 2
            subplot(1,2,2);
            imshow(I);
            set(gcf, 'Position', get(0, 'Screensize'));
            %
            roi = images.roi.Polyline(gca,'Position',ROI1,'MarkerSize',4);
            % 
            addlistener(roi,'MovingROI',@allevents);
            pause;
            %
            load('ROI1.mat','ROI1');
            close all;
            %
            figure;
            set(gcf, 'Position', get(0, 'Screensize'));
            % subplot 1
            blank_mask = zeros(image_size,image_size);    
            mask = roipoly(blank_mask,ROI1(:,1),ROI1(:,2));
            subplot(1,2,1)
            imshow(mask);
            % subplot 2
            subplot(1,2,2)
            imshow(I);
            hold on;
            plot(ROI1(:,1),ROI1(:,2),'g');
            sgtitle(tit);
            hold off;
            message = sprintf('Are you satisfied with the new mask?');
            answer = MFquestdlg([0.4 0.7],message,'MaskEdit', 'Yes', 'No', 'Yes');
            %
            if strcmp(answer,'Yes')
                figure;
                set(gcf,'PaperUnits','inches','PaperPosition',[0 0 resolution_factor resolution_factor]);
                axes('Units', 'normalized', 'Position', [0 0 1 1]);
                blank_mask = zeros(image_size,image_size);
                mask = roipoly(blank_mask,ROI1(:,1),ROI1(:,2));
                imshow(mask);
                imwrite(mask,strcat(newmask_path, '\', data_mask(i).name),'png')
                if exist(fullfile(Reject_path, data_mask(i).name), 'file')
                    delete(fullfile(Reject_path, data_mask(i).name));
                end
                if exist(fullfile(Unsure_path, data_mask(i).name), 'file')
                    delete(fullfile(Unsure_path, data_mask(i).name));
                end
                close all;
                break;
            elseif strcmp(answer,'No')
                %
            else
                return;
            end
        end
    elseif  strcmp(answer,'Reject')
        message = sprintf('Would you like to make a New Mask, Unsure or Reject');
        answer = MFquestdlg([0.4 0.7],message,'MaskEdit', 'New Mask', 'Unsure', 'Reject', 'New Mask');
        if  strcmp(answer,'New Mask')
            answer='No';
            new1=[];
            new2=[];
            while(strcmp(answer, 'No'))
                close all;
                figure;
                subplot(1,2,1)
                imshow(imcomplement(I))
                subplot(1,2,2)
                imshow(I)
                set(gcf, 'Position', get(0, 'Screensize'));
                sgtitle(tit);
                if ~isempty(new1)
                    new1=drawpolyline('Position',tempnew1);
                    pause;
                else
                    new1 = drawpolyline('Color','r');
                end
                %
                tempnew1=new1.Position;
                figure;
                set(gcf, 'Position', get(0, 'Screensize'));
                % subplot 1
                subplot(1,2,1)
                blank_mask = zeros(image_size,image_size);    
                mask = roipoly(blank_mask,tempnew1(:,1),tempnew1(:,2));
                imwrite(mask,strcat(newmask_path, '\', data_mask(i).name),'png')
                % subplot 2
                subplot(1,2,2)
                imshow(I);
                hold on;
                plot(tempnew1(:,1),tempnew1(:,2),'g');
                sgtitle(tit);
                hold off;
                message = sprintf('Are you satisfied with the new mask?');
                answer = MFquestdlg([0.4 0.7],message,'MaskEdit', 'Yes', 'No','Yes');
                if strcmp(answer,'Yes')
                    close all;
                    figure;
                    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 resolution_factor resolution_factor]);
                    axes('Units', 'normalized', 'Position', [0 0 1 1])
                    blank_mask = zeros(image_size,image_size);    
                    mask = roipoly(blank_mask,tempnew1(:,1),tempnew1(:,2));
                    imshow(mask);
                    imwrite(mask,strcat(newmask_path, '\', data_mask(i).name),'png')
                    if exist(fullfile(Reject_path, data_mask(i).name), 'file')
                        delete(fullfile(Reject_path, data_mask(i).name));
                    end
                    if exist(fullfile(Unsure_path, data_mask(i).name), 'file')
                        delete(fullfile(Unsure_path, data_mask(i).name));
                    end
                    break
                elseif strcmp(answer,'No')
                    %
                else
                    return;
                end
            end
        elseif strcmp(answer,'Unsure')
            imwrite(M, strcat(Unsure_path, '\', data_mask(i).name));
            if exist(fullfile(newmask_path, data_mask(i).name), 'file')
                delete(fullfile(newmask_path, data_mask(i).name));
            end
            if exist(fullfile(Reject_path, data_mask(i).name), 'file')
                delete(fullfile(Reject_path, data_mask(i).name));
            end
        elseif strcmp(answer,'Reject')
            imwrite(M, strcat(Reject_path, '\', data_mask(i).name));
            if exist(fullfile(newmask_path, data_mask(i).name), 'file')
                delete(fullfile(newmask_path, data_mask(i).name));
            end
            if exist(fullfile(Unsure_path, data_mask(i).name), 'file')
                delete(fullfile(Unsure_path, data_mask(i).name));
            end
        else
            break;
        end
    else
        break;
    end
    % save page index
    save('PageNumber.mat','i')
end