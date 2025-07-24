% Step 1: 
data_folder ="D:\Desktop\Attachment 1";
image_files = [dir(fullfile(data_folder, '*.png')); dir(fullfile(data_folder, '*.jpg'))];


color_shift_data = [];
brightness_data = [];
blur_data = [];
file_names = {};


for i = 1:length(image_files)
   
    img_path = fullfile(image_files(i).folder, image_files(i).name);
    img = imread(img_path);
    
   
    [color_shift, brightness, blur] = analyze_degradation(img);
    
    
    color_shift_data = [color_shift_data; color_shift];
    brightness_data = [brightness_data; brightness];
    blur_data = [blur_data; blur];
    file_names{end+1} = image_files(i).name;
    
    
    depth_map = linspace(0, 1, size(img, 1))' * ones(1, size(img, 2)); 
    beta_r = 0.8; beta_g = 0.6; beta_b = 0.4;
    transmission_r = transmission_model(depth_map, beta_r);
    transmission_g = transmission_model(depth_map, beta_g);
    transmission_b = transmission_model(depth_map, beta_b);
    
    
    degraded_img_color = color_cast_degradation(img, transmission_r, transmission_g, transmission_b);
    
    
    alpha = 0.5;
    degraded_img_low_light = low_light_degradation(img, depth_map, alpha);
    
    
    sigma = 5;
    degraded_img_blur = blur_degradation(img, sigma);
    
    
    figure;
    subplot(2, 2, 1), imshow(img), title(['original image: ', image_files(i).name]);
    subplot(2, 2, 2), imshow(degraded_img_color), title('Color shift degradation');
    subplot(2, 2, 3), imshow(degraded_img_low_light), title('Low light degradation');
    subplot(2, 2, 4), imshow(degraded_img_blur), title('Fuzzy degradation');
end


T = table(file_names', color_shift_data(:, 1), color_shift_data(:, 2), color_shift_data(:, 3), ...
    brightness_data, blur_data, ...
    'VariableNames', {'Image', 'ColorShift_R', 'ColorShift_G', 'ColorShift_B', 'Brightness', 'Blur'});


output_excel = 'D:\question.xlsx';
writetable(T, output_excel);
disp(['The analysis results have been saved to: ', output_excel]);

%% Step 2
figure;

subplot(1, 2, 1);
plot(1:length(image_files), brightness_data, 'o-', 'DisplayName', 'Brightness');
title('brightness distribution');
xlabel('image index');
ylabel('brightness');
grid on;
legend;


subplot(1, 2, 2);
plot(1:length(image_files), blur_data, 's-', 'Color', 'r', 'DisplayName', 'Blur Level');
title('Fuzzy degree distribution');
xlabel('image index');
ylabel('fog-level');
grid on;
legend;




function t = transmission_model(d, beta)
    %  t = exp(-beta * d)
    t = exp(-beta .* d);
end


function [color_shift, brightness, blur] = analyze_degradation(img)

    img = im2double(img);
    
 
    r = img(:,:,1);
    g = img(:,:,2);
    b = img(:,:,3);
    gray = rgb2gray(img);
    
   
    mean_r = mean(r(:));
    mean_g = mean(g(:));
    mean_b = mean(b(:));
    beta_r = abs(mean_r - mean_g);
    beta_g = abs(mean_g - mean_b);
    beta_b = abs(mean_b - mean_r);
    color_shift = [beta_r, beta_g, beta_b];
    
  
    brightness = mean(gray(:));
    
  
    laplacian = fspecial('laplacian');
    laplacian_img = imfilter(gray, laplacian, 'replicate');
    blur = var(laplacian_img(:));
end


function degraded_img = color_cast_degradation(img, transmission_r, transmission_g, transmission_b)
    img = im2double(img);
    degraded_img(:,:,1) = img(:,:,1) .* transmission_r;
    degraded_img(:,:,2) = img(:,:,2) .* transmission_g;
    degraded_img(:,:,3) = img(:,:,3) .* transmission_b;
    degraded_img = im2uint8(degraded_img);
end


function degraded_img = low_light_degradation(img, depth, alpha)
    img = im2double(img);
    light_attenuation = exp(-alpha * depth);
    degraded_img = img .* light_attenuation;
    degraded_img = im2uint8(degraded_img);
end


function degraded_img = blur_degradation(img, sigma)
    img = im2double(img);
    h = fspecial('gaussian', [5 5], sigma);
    degraded_img = imfilter(img, h, 'replicate');
    degraded_img = im2uint8(degraded_img);
end
