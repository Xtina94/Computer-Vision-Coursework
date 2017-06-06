%% Section 1 : Read the images
clear all;
close all;


num_images = 6;
num_iteration = 50;
angle = 29;

im = cell(num_images,1);
des = cell(num_images,1);
loc = cell(num_images,1);

match = cell(num_images-1,1);
num = zeros(num_images-1,1);

M = cell(num_images-1,1);
T = cell(num_images-1,1);

deltaX = zeros(num_images-1,1);
deltaY = zeros(num_images-1,1);

util = Util;

%1
for i=1:num_images
    %1
    im_i = imread(strcat('../Ge/i', num2str(i), 'scale.jpg'));
    %2
    im_pi = util.projectIC(im_i,angle);
    [im{i,1}, des{i,1}, loc{i,1}] = sift(im_pi);
end

dim_x = size(im_i,2);
dim_y = size(im_i,1);

%util.showkeys(im{1,1},loc{1,1});
%% Section 2 : Compute transformation
mean_dx = zeros(1,num_images-1);
mean_dy = zeros(1,num_images-1);
threshold = 0.7;

for i=1:num_images-1
    im_a = im{i,1}; des_a = des{i,1}; loc_a = loc{i,1};
    im_b = im{i+1,1}; des_b = des{i+1,1}; loc_b = loc{i+1,1};
    %3
    [match{i,1}, num(i)] = util.match2(im_a, des_a, loc_a, im_b, des_b, loc_b, threshold);
end

% %corr is the matrix of correspondences
%   i: index of loc_a
%   match(i): index of loc_b
matches = cell(num_images-1,1);
for z = 1:num_images-1
    corr = zeros(num(z),2);
    j = 1;
    for k=1:size(match{z,1},2)
            if(match{z,1}(k) > 0)
                    corr(j,1) = k;                %   i: index of loc_a
                    corr(j,2) = match{z,1}(k);    %   match(i): index of loc_b
                    j = j+1;
            end
    end
    matches{z,1} = corr;
end
%% 
for z=1:num_images-1 %for each image
    [M{z},T{z}] = util.compute_M(z, num_iteration, num, matches, loc);
end

for z = 1:num_images-1
    pick = randi(num(z));
    x = loc{z}(matches{z}(pick,1),2);
    y = loc{z}(matches{z}(pick,1),1);
    UV = M{z}*[x;y]+T{z};
    deltaX(z) = (dim_x - x) + UV(1);
    deltaY(z) = y - UV(2);
end

%%

% sliding images vertically.
dy_pos = round(sum(deltaY(deltaY > 0)));
dy_neg = round(sum(deltaY(deltaY < 0)));

im{1,1} = [zeros(abs(dy_neg),dim_x); im{1,1}; zeros(dy_pos,dim_x)];


for i=2:num_images
    dy = round(sum(deltaY(1:i-1)));
        A = [zeros(abs(dy_neg)+dy,dim_x); im{i,1}; zeros(dy_pos-dy,dim_x)];
    im{i,1} = A;
end

pan_img = im{1,1};

for i=1:num_images-1
    im_i = im{i+1,1};
    pan_img = [pan_img(:,1:end-deltaX(i)) im_i];
end

imshow(pan_img);










