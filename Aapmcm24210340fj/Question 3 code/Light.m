%% Step 1: 

folderPath = "D:\Desktop\classified\Light"; 
resultFilePath = '2.xlsx'; 



files_jpg = dir(fullfile(folderPath, '*.jpg'));
files_png = dir(fullfile(folderPath, '*.png'));
files = [files_jpg; files_png]; 
numImages = length(files);


results = cell(numImages, 7);
results(:, 1) = {files.name}; 

%% Step 2: 
for i = 1:numImages
    fprintf('Processing image %s\n', files(i).name);
    imagePath = fullfile(files(i).folder, files(i).name);
    img = imread(imagePath);
    
    
    img = im2double(img);
    reference = 0.5 * ones(size(img)); 

    
    psnr_before = calculatePSNR(img, reference);
    uciqe_before = calculateUCIQE(img);
    uiqm_before = calculateUIQM(img);
    
    
   
 
    enhancedImg = enhanceBrightness(img);
    

   
    psnr_after = calculatePSNR(enhancedImg, reference);
    uciqe_after = calculateUCIQE(enhancedImg);
    uiqm_after = calculateUIQM(enhancedImg);
    
   
    results{i, 2} = psnr_before;
    results{i, 3} = uciqe_before;
    results{i, 4} = uiqm_before;
    results{i, 5} = psnr_after;
    results{i, 6} = uciqe_after;
    results{i, 7} = uiqm_after;

    
    [~, name, ext] = fileparts(files(i).name);
    enhancedImagePath = fullfile(folderPath, ['enhanced_' name ext]);
    imwrite(uint8(enhancedImg * 255), enhancedImagePath);
    
   
    fprintf('Processed %d of %d images.\n', i, numImages);
end


writeResultsToExcel(results, resultFilePath);
fprintf('All images are processed and the results are saved to %s\n', resultFilePath);

%% Step 3: 

function enhancedImg = enhanceBrightness(img)
    imgHSV = rgb2hsv(img);
    V = imgHSV(:,:,3);
    
   
    V = adapthisteq(V, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
    
   
    imgHSV(:,:,3) = V;
    enhancedImg = hsv2rgb(imgHSV);
   
end


%% Step 4:

function psnr_value = calculatePSNR(original, reference)
    mse = mean((original - reference).^2, 'all');
    if mse == 0
        psnr_value = Inf;
    else
        maxVal = 1; 
        psnr_value = 10 * log10((maxVal^2) / mse);
    end
end


function uciqe_value = calculateUCIQE(img)
    imgLab = rgb2lab(img);
    L = imgLab(:,:,1);
    A = imgLab(:,:,2);
    B = imgLab(:,:,3);
    chroma = sqrt(A.^2 + B.^2);
    
    
    uciqe_value = std(chroma(:)) + std(L(:)) + mean(L(:)) / (mean(chroma(:)) + 1e-6);
end


function uiqm_value = calculateUIQM(img)
    imgHSV = rgb2hsv(img);
    H = imgHSV(:,:,1);
    S = imgHSV(:,:,2);
    V = imgHSV(:,:,3);
    
    
    uiqm_value = 0.5 * std(H(:)) + 0.25 * std(S(:)) + 0.25 * mean(V(:));
end

%% Step 5: 
function writeResultsToExcel(results, resultFilePath)
    
    T = table(results(:, 1), results(:, 2), results(:, 3), results(:, 4), ...
              results(:, 5), results(:, 6), results(:, 7), ...
              'VariableNames', {'FileName', 'PSNR', 'UCIQE', 'UIQM', 'PSNR-IM', 'UCIQE-IM', 'UIQM-IM'});
    
    
    writetable(T, resultFilePath);
end
