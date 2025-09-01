clc; 

% %% Test single file
img1 = double(HR_RHSI); img2 = double(R_hsi);
% Using % for SLNET and AUNET (No not use global norm for these methods)
img1 = (img1 - min(img1(:))) ./ (max(img1(:)) - min(img1(:)));
img2 = (img2 - min(img2(:))) ./ (max(img2(:)) - min(img2(:)));

%% Point normalization
[H, W, band] = size(img1);
psnr_all = zeros(band, 1); ssim_all = zeros(band, 1);

for i = 1:band
    temp_img1 = squeeze(img1(:, :, i)); temp_img2 = squeeze(img2(:, :, i));
    %temp_img1 = (temp_img1 - min(temp_img1(:))) ./ (max(temp_img1(:)) - min(temp_img1(:)));
    %temp_img2 = (temp_img2 - min(temp_img2(:))) ./ (max(temp_img2(:)) - min(temp_img2(:)));

    psnr_all(i, 1) = psnr(temp_img2, temp_img1);
    ssim_all(i, 1) = ssim(temp_img2, temp_img1);
end
mean(psnr_all)
mean(ssim_all)

%% Test all poit SAM
img1 = double(HR_RHSI); img2 = double(R_hsi);
[H, W, band] = size(img1);
img1 = reshape(img1, [H*W, band]);  img2 = reshape(img2, [H*W, band]);
SAM_all = zeros(H*W, 1);

for i = 1:H*W
    temp_spectral_1 = squeeze(img1(i, :)); 
    temp_spectral_2 = squeeze(img2(i, :));
    temp_spectral_1 = (temp_spectral_1 - min(temp_spectral_1(:))) ./ (max(temp_spectral_1(:)) - min(temp_spectral_1(:)));
    temp_spectral_2 = (temp_spectral_2 - min(temp_spectral_2(:))) ./ (max(temp_spectral_2(:)) - min(temp_spectral_2(:)));
    dot_product = dot(temp_spectral_1, temp_spectral_2);
    norm_x = norm(temp_spectral_1);
    norm_y = norm(temp_spectral_2);

    SAM_all(i, 1) = acos(dot_product / (norm_x * norm_y));
end
mean(SAM_all)

clear temp_spectral_1; clear temp_spectral_2; clear dot_product; clear norm_x;  clear norm_y;
clear img1; clear img2; clear temp_img1; clear temp_img2;
clear H; clear  W; clear band; clear i;