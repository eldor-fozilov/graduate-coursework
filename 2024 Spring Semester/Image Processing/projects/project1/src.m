
%---------------------
% Problem 1
%---------------------

% step 1 (download and read the image)
a = imread("GrangerRainbow.png");
imshow(a)
title("Original Image","FontSize", 16);

image = a;

% step 2
% check min and max values
display(max(image(:)));
display(min(image(:)));
% convert to double type and normalize
norm_image = double(image) / 255.0;
% convert nonlinear sRGB to linear RGB using gamma 2.2
gamma = 2.2;
lin_image = norm_image.^gamma;
% show the output image
imshow(lin_image)
title("Linear sRGB", "FontSize", 16);

% step 3 (convert to linear p3 RGB)
trans_matrix = [0.8225 0.1775 0.0001;
                0.0331 0.9668 0.0000;
                0.0171 0.0724 0.9105];

lin_image_p3 = reshape(lin_image, [], 3) * trans_matrix';
lin_image_p3 = reshape(lin_image_p3, size(lin_image));

% step 4
% convert the linear p3 RGB image to
% nonlinear p3 RGB image using 1/2.2 gamma
inv_gamma = 1/2.2;
nonlin_image_p3 = lin_image_p3.^inv_gamma;
% show the image
imshow(nonlin_image_p3);
title("Nonlinear p3 RGB",'FontSize', 16);

% step 5
% find and display the difference between the nonlinear sRGB and nonlinear p3 RGB images 
imshowpair(norm_image, nonlin_image_p3);

% Convert linear sRGB and p3 RGB images into XYZ color space and then to Lab colorspace
trans_matrix_s = [0.4124 0.3576 0.1805;
                  0.2126 0.7152 0.0722;
                  0.0193 0.1192 0.9505];
trans_matrix_p = [0.4866 0.2657 0.1982;
                  0.2290 0.6917 0.0793;
                  0.0000 0.0451 1.0439];

s_xyz = reshape(lin_image, [], 3) * trans_matrix_s';
s_xyz = reshape(s_xyz, size(lin_image));

p3_xyz = reshape(lin_image_p3, [], 3) * trans_matrix_p';
p3_xyz = reshape(p3_xyz, size(lin_image_p3));

lab_rgb = xyz2lab(s_xyz);
lab_p3 = xyz2lab(p3_xyz);

% find the average and max of color difference
[Ls, as, bs] = imsplit(lab_rgb);
[Lp, ap, bp] = imsplit(lab_p3);
deltaE = sqrt((Ls - Lp).^2 + (as - ap).^2 + (bs - bp).^2);
% display mean and max color distances
disp(mean(deltaE(:)));
disp(max(deltaE(:)));

% step 6
% convert the linear sRGB image to 10 bit precision
bit_precision_scale = 2.^10;

bit10_lin_image = round(bit_precision_scale*lin_image);

% convert the conversion matrix to 10 bit precision
bit10_H = round(bit_precision_scale * trans_matrix);

% convert linear sRGB to linear p3 RGB
bit10_lin_image_reshaped = reshape(bit10_lin_image, [], 3);
bit10_lin_image_p3 = round(bit10_lin_image_reshaped * bit10_H'./bit_precision_scale');
bit10_lin_image_p3 = reshape(bit10_lin_image_p3, size(bit10_lin_image));

% convert the 10 bit linear s RGB to double format
bit10_double_lin_image = bit10_lin_image./bit_precision_scale;
bit10_s_xyz = reshape(bit10_double_lin_image, [], 3);
bit10_s_xyz = reshape(bit10_s_xyz * trans_matrix_s', size(bit10_double_lin_image));
bit10_lab_rgb = xyz2lab(bit10_s_xyz);

% convert the 10 bit linear p3 RGB to double format
bit10_double_lin_image_p3 = bit10_lin_image_p3./bit_precision_scale; 
bit10_p3_xyz = reshape(bit10_double_lin_image_p3, [], 3);
bit10_p3_xyz = reshape(bit10_p3_xyz * trans_matrix_p', size(bit10_double_lin_image_p3));
bit10_lab_p3 = xyz2lab(bit10_p3_xyz);

% calculate deltaE and its mean and max

[Ls, as, bs] = imsplit(bit10_lab_rgb);
[Lp, ap, bp] = imsplit(bit10_lab_p3);
bit10_deltaE = sqrt((Ls - Lp).^2 + (as - ap).^2 + (bs - bp).^2);

disp(mean(bit10_deltaE(:)));
disp(max(bit10_deltaE(:)));

% step 7

% convert the linear sRGB image to 8 bit precision
bit_precision_scale = 2.^8;

bit8_lin_image = round(bit_precision_scale*lin_image);

% convert the conversion matrix to 10 bit precision
bit8_H = round(bit_precision_scale * trans_matrix);

% convert linear sRGB to linear p3 RGB
bit8_lin_image_reshaped = reshape(bit8_lin_image, [], 3);
bit8_lin_image_p3 = round(bit8_lin_image_reshaped * bit8_H'./bit_precision_scale);
bit8_lin_image_p3 = reshape(bit8_lin_image_p3, size(bit8_lin_image));

% convert the 10 bit linear s RGB to double format
bit8_double_lin_image = bit8_lin_image./bit_precision_scale;
bit8_s_xyz = reshape(bit8_double_lin_image, [], 3);
bit8_s_xyz = reshape(bit8_s_xyz * trans_matrix_s', size(bit8_double_lin_image));
bit8_lab_rgb = xyz2lab(bit8_s_xyz);

% convert the 10 bit linear p3 RGB to double format
bit8_double_lin_image_p3 = bit8_lin_image_p3./bit_precision_scale; 
bit8_p3_xyz = reshape(bit8_double_lin_image_p3, [], 3);
bit8_p3_xyz = reshape(bit8_p3_xyz * trans_matrix_p', size(bit8_double_lin_image_p3));
bit8_lab_p3 = xyz2lab(bit8_p3_xyz);

% calculate deltaE and its mean and max
[Ls, as, bs] = imsplit(bit8_lab_rgb);
[Lp, ap, bp] = imsplit(bit8_lab_p3);
bit8_deltaE = sqrt((Ls - Lp).^2 + (as - ap).^2 + (bs - bp).^2);

disp(mean(bit8_deltaE(:)));
disp(max(bit8_deltaE(:)));


%---------------------
% Problem 2
%---------------------

% 2.1

C = 2; % conversion 

% the sample-and-hold filter
h_sah = ones(1, C);

% the linear interpolation filter
h_lin = zeros(1, 2*C+1);
for n = -C:C
    h_lin(n+C+1) = sinc(n/2);
end

% the cubic convolution filter
a = -0.5;
h_cc = zeros(1, 4*C+1);
for n = -2*C:2*C
    h_cc(n+2*C+1) = (a+2)*abs(n)^3 - (a+3)*abs(n)^2 + 1;
    h_cc(n+2*C+1) = h_cc(n+2*C+1) * sinc(n/2*(1-a));
end 

% plot the impulse response
figure;
stem(h_sah);
xlabel('Samples')
title('Sample-and-Hold Interpolation Filter Impulse Response');

% the magnitude response
figure;
freqz(h_sah, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Sample-and-Hold Interpolation Filter Magnitude Response');

% plot the impulse response
figure;
stem(h_lin);
xlabel('Samples');
title('Linear Interpolation Filter Impulse Response');

% plot the magnitude response
figure;
freqz(h_lin, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Linear Interpolation Filter Magnitude Response');

% plot the impulse response
figure;
stem(h_cc);
xlabel('Samples');
title('Cubic Convolutional Interpolation Filter Impulse Response');

% plot the magnitude response
figure;
freqz(h_cc, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Cubic Convolutional Interpolation Filter Magnitude Response');

% 2.2

C = 3; % conversion

% the sample-and-hold filter
h_sah = ones(1, C);

% the linear interpolation filter
h_lin = zeros(1, 2*C+1);
for n = -C:C
    h_lin(n+C+1) = sinc(n/2);
end

% the cubic convolution filter
a = -0.5;
h_cc = zeros(1, 4*C+1);
for n = -2*C:2*C
    h_cc(n+2*C+1) = (a+2)*abs(n)^3 - (a+3)*abs(n)^2 + 1;
    h_cc(n+2*C+1) = h_cc(n+2*C+1) * sinc(n/2*(1-a));
end 

% plot the impulse response
figure;
stem(h_sah);
xlabel('Samples')
title('Sample-and-Hold Interpolation Filter Impulse Response');

% the magnitude response
figure;
freqz(h_sah, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Sample-and-Hold Interpolation Filter Magnitude Response');

% plot the impulse response
figure;
stem(h_lin);
xlabel('Samples');
title('Linear Interpolation Filter Impulse Response');

% plot the magnitude response
figure;
freqz(h_lin, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Linear Interpolation Filter Magnitude Response');

% plot the impulse response
figure;
stem(h_cc);
xlabel('Samples');
title('Cubic Convolutional Interpolation Filter Impulse Response');

% plot the magnitude response
figure;
freqz(h_cc, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Cubic Convolutional Interpolation Filter Magnitude Response');


% 2.3

load('a.mat');
figure;
plot(a)
title("Vector A")

% 2.4

% upsample by 3 using sample-and-hold filter
sah_up_sig1 = resample(a, 3, 1);

% interpolate with a third band filter
sah_f1 = fir1(2, 1/3, 'high');
inter_sah_sig1 = conv(sah_up_sig1, sah_f1, 'same');

% normalize DC gain of the filter
dc_gain_f1 = sum(sah_f1);
sah_f1 = sah_f1 / dc_gain_f1;
inter_sah_sig1 = inter_sah_sig1 / dc_gain_f1;

% anti-alias with a half band filter
sah_f2 = fir1(2, 0.5, 'low');
anti_aliased_sah_sig1 = conv(inter_sah_sig1, sah_f2, 'same');

% normalize DC gain of the filter
dc_gain_f2 = sum(sah_f2);
sah_f2 = sah_f2 / dc_gain_f2;
anti_aliased_sah_sig1 = anti_aliased_sah_sig1 / dc_gain_f2;

% downsample by 2
sah_down_sig1 = downsample(anti_aliased_sah_sig1, 2);

% plot the results
figure;
plot(sah_down_sig1);
title('Sample-and-Hold: Signal scaled by 3/2');

% 2.5

% upsample by 2
sah_ups_sig2 = upsample(sah_down_sig1, 2);

% interpolate with a half band filter
sah_f3 = fir1(2, 0.5, 'low');
inter_sah_sig2 = conv(sah_ups_sig2, sah_f3, 'same');

% normalize DC gain of the filter
dc_gain_f3 = sum(sah_f3);
sah_f3 = sah_f3 / dc_gain_f3;
inter_sah_sig2 = inter_sah_sig2 / dc_gain_f3;

% anti-alias with a third band filter
sah_f4 = fir1(2, 1/3, 'low');
anti_aliased_sah_sig2 = conv(inter_sah_sig2, sah_f4, 'same');

% normalize DC gain of the filter
dc_gain_f4 = sum(sah_f4);
sah_f4 = sah_f4 / dc_gain_f4;
anti_aliased_sah_sig2 = anti_aliased_sah_sig2 / dc_gain_f4;

% downsample by 3
sah_down_sig2 = downsample(anti_aliased_sah_sig2, 3);

% plot the results
figure;
plot(sah_down_sig2);
title('Sample-and-Hold: Signal scaled by 2/3');

% 2.6

% compute the absolute error between scaled signals and original signal
error = a - sah_down_sig2;

% compensate for the group delay caused by filtering
group_delay = mean(grpdelay(sah_f1)) + mean(grpdelay(sah_f2)) + mean(grpdelay(sah_f3)) + mean(grpdelay(sah_f4));
error_compensated = error(group_delay+1:end);

% plot the error
figure;
plot(error);
title("Error Between Scaled and Original Signals")

% compute the mean squared error (MSE)
MSE = mean(error_compensated .^ 2);
disp(MSE);


%---------------------
% Problem 3
%---------------------

% 3.1

wvlet = 'db4'; % or 'haar' for Haar wavelet
[c, d, e, f] = wfilters(wvlet);

% c

% plot the impulse response
figure;
stem(c);
xlabel('Samples')
title('Analysis Filter Impulse Response (c)');

% the magnitude response
figure;
freqz(c, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Analysis Filter Magnitude Response (c)');

% d
% plot the impulse response
figure;
stem(d);
xlabel('Samples')
title('Analysis Filter Impulse Response (d)');

% the magnitude response
figure;
freqz(d, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Analysis Filter Magnitude Response (d)');

% e
% plot the impulse response
figure;
stem(e);
xlabel('Samples')
title('Analysis Filter Impulse Response (e)');

% the magnitude response
figure;
freqz(e, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Analysis Filter Magnitude Response (e)');

% f
% plot the impulse response
figure;
stem(f);
xlabel('Samples')
title('Analysis Filter Impulse Response (f)');

% the magnitude response
figure;
freqz(f, 1);
xlabel('Normalized Frequency');
ylabel('Magnitude (dB)');
title('Analysis Filter Magnitude Response (f)');

% 3.2

load('a.mat');
% analysis filterbank
high_level = conv(a, c, 'valid');
low_level = conv(a, d, 'valid');
% synthesis filterbank
reconst = conv(high_level, e, 'full') + conv(low_level, f, 'full');
reconst = reconst / 2;

% plot input and output
figure;
subplot(3,1,1);
plot(a);
title('Original Signal');

subplot(3,1,2);
plot(reconst);
title('Reconstructed Signal');

subplot(3,1,3);
plot(a - reconst);
title('Reconstruction Error');

% calculate MSE
MSE = mean((a - reconst).^2);
disp(MSE);

% 3.3

% level 1

% analysis filterbank
high_level = conv(reconst, c, 'valid');
low_level = conv(reconst, d, 'valid');
% synthesis filterbank
reconst2 = conv(high_level, e, 'full') + conv(low_level, f, 'full');
reconst2 = reconst2 / 2;

% plot input and output
figure;
subplot(3,1,1);
plot(reconst);
title('Original Signal (Level 1)');

subplot(3,1,2);
plot(reconst2);
title('Reconstructed Signal (Level 1)');

subplot(3,1,3);
plot(reconst - reconst2);
title('Reconstruction Error (Level 1)');

% calculate MSE
MSE = mean((reconst - reconst2).^2);
disp(MSE);

% level 2

% analysis filterbank
high_level = conv(reconst2, c, 'valid');
low_level = conv(reconst2, d, 'valid');
% synthesis filterbank
reconst3 = conv(high_level, e, 'full') + conv(low_level, f, 'full');
reconst3 = reconst3 / 2;

% plot input and output
figure;
subplot(3,1,1);
plot(reconst2);
title('Original Signal (Level 2)');

subplot(3,1,2);
plot(reconst3);
title('Reconstructed Signal (Level 2)');

subplot(3,1,3);
plot(reconst2 - reconst3);
title('Reconstruction Error (Level 2)');

% calculate MSE
MSE = mean((reconst2 - reconst3).^2);
disp(MSE);

% 3.4

image = imread('barbara.png');
[LL, LH, HL, HH] = dwt2(image, wvlet);
reconst_image = idwt2(LL, LH, HL, HH, wvlet);

% calculate error and MSE
error_image = double(image) - reconst_image;

% plot the error
figure;
plot(error_image);
title("Image Reconstruction Error");

MSE_image = mean(error_image(:).^2);
disp(MSE_image);

% 3.5

[LL, LH, HL, HH] = dwt2(image, wvlet, 3);
min_val = min([LL(:); LH(:); HL(:); HH(:)]);
max_val = max([LL(:); LH(:); HL(:); HH(:)]);
delta = (max_val - min_val) / (2.^8 - 1);

% quantize wavelet coefficients
quantized_LL = delta * round(LL / u);
quantized_LH = delta * round(LH / u);
quantized_HL = delta * round(HL / u);
quantized_HH = delta * round(HH / u);

% apply IDWT
reconst_quantized_image = idwt2(quantized_LL, quantized_LH, quantized_HL, quantized_HH, wvlet);

% calculate error and MSE
error_quantized_image = double(image) - reconst_quantized_image;

% plot the error
figure;
plot(error_quantized_image);
title("Quantized Image Reconstruction Error");

MSE_quantized_image = mean(error_quantized_image(:).^2);
disp(MSE_quantized_image);
