
% ----------------- Problem 1: PCA ----------------

% 1.1

% load the image, convert it to YCbCr
img_rgb = imread("k03.bmp");
img_ycbcr = rgb2ycbcr(img_rgb);
img = double(img_ycbcr(:, :, 1)); % Y-channel Image


% display the Y-channel image
figure;
subplot(2,2,1);
imshow(uint8(img)); title("Y-channel Image");

% 1.2

% a)

block_size = 8;
[H,W] = size(img);
K = 4; % number of lowest freq components to keep

% Apply 2D DCT
img_dct = zeros(size(img));
for i = 1:block_size:H
    for j = 1:block_size:W

        block = img(i:i+block_size-1, j:j+block_size-1);
        dct_block = dct2(block);
        img_dct(i:i+block_size-1, j:j+block_size-1) = dct_block;
    end
end

% Select lowest frequency components in zigzag scan order
for row = 1:block_size:H
    for col = 1:block_size:W
        block = img_dct(row:row+block_size-1, col:col+block_size-1);
        zigzag = zigzagScan(block);
        sortedZigzag = sort(zigzag, 'descend');
        threshold = sortedZigzag(K+1);
        block(abs(block) < threshold) = 0;
        img_dct(row:row+block_size-1, col:col+block_size-1) = block;
    end
end

% Apply block-wise 2D IDCT to reconstruct the image
img_reconst = zeros(size(img));
for i = 1:block_size:H
    for j = 1:block_size:W
        dct_block = img_dct(i:i+block_size-1, j:j+block_size-1);
        block_reconst = idct2(dct_block);
        img_reconst(i:i+block_size-1, j:j+block_size-1) = block_reconst;
    end
end

% Initialize the DCT basis functions
dct_basis = zeros(block_size, block_size, block_size, block_size);

% Compute the DCT basis functions
for u = 0:block_size-1
    for v = 0:block_size-1
        for x = 0:block_size-1
            for y = 0:block_size-1
                dct_basis(x+1,y+1,u+1,v+1) = cos((2*x+1)*u*pi/(2*block_size)) * cos((2*y+1)*v*pi/(2*block_size));
            end
        end
    end
end

% Display the reconstructed image
subplot(1,2,2); imshow(uint8(img_reconst));
title('Reconstructed Image');

% Display the absolute error
abs_error = abs(img - img_reconst);
mean_error = round(mean(abs_error(:)), 2);
subplot(2,2,3);
imshow(abs_error, []); title(['Absolute Error Plot (MAE: ', num2str(mean_error), ')']);

% Define the zigzag order for an 8x8 block
zigzag_order = [
     1  2  6  7 15 16 28 29
     3  5  8 14 17 27 30 43
     4  9 13 18 26 31 42 44
    10 12 19 25 32 41 45 54
    11 20 24 33 40 46 53 55
    21 23 34 39 47 52 56 61
    22 35 38 48 51 57 60 62
    36 37 49 50 58 59 63 64];

% Display the K 2D DCT basis
figure;
for k = 1:K
    [u, v] = find(zigzag_order == k);
    subplot(ceil(sqrt(K)), ceil(sqrt(K)), k);
    imagesc(dct_basis(:, :, k));
    colormap(gray);
    title(['Basis Function ', num2str(k), ' (', num2str(u), ',', num2str(v), ')']);
end

function output = zigzagScan(matrix)
    % Zigzag scan order
    numRows = size(matrix, 1);
    numCols = size(matrix, 2);
    output = zeros(1, numRows * numCols);
    row = 1;
    col = 1;
    idx = 1;
    while row <= numRows && col <= numCols
        output(idx) = matrix(row, col);
        if mod(row + col, 2) == 0 % Even sum, go up
            if col == numCols
                row = row + 1;
            elseif row == 1
                col = col + 1;
            else
                row = row - 1;
                col = col + 1;
            end
        else % Odd sum, go down
            if row == numRows
                col = col + 1;
            elseif col == 1
                row = row + 1;
            else
                row = row + 1;
                col = col - 1;
            end
        end
        idx = idx + 1;
    end
end


% b)

p = 0.9; % correlation
M = 6; % number of PCAs to keep

% Covariance matrix
C = zeros(block_size, block_size);

% Calculate the covariance matrix using the AR 1 model
for i = 1:block_size
    for j = 1:block_size
        C(i, j) = p^(abs(i - j));
    end
end

% Display the covariance matrix
figure;
imagesc(C);
colormap(gray);
title('Covariance Matrix');

% Find the eigenvectors and eigenvalues of the covariance matrix
[V, D] = eig(C);


% Find the outer products of the eigenvectors
outer_products = zeros(block_size, block_size,block_size);
for i = 1:block_size
    outer_products(:,:,i) = V(:,i) * V(:,i)';
end

% Extract and sort eigenvalues in descending order
eigenvalues = diag(D);
[eigenvalues_sorted, idx] = sort(eigenvalues, 'descend');

% Sort eigenvectors based on the sorted eigenvalues
V_sorted = V(:, idx);

% Plot the eigenvalues in descending order
figure;
plot(eigenvalues_sorted, 'bo-', 'LineWidth', 2);
title('Sorted Eigenvalues');
xlabel('Eigenvalue Index');
ylabel('Eigenvalue');

% Display the M biggest 2D principal components
figure;
for i = 1:M
    subplot(ceil(sqrt(M)), ceil(sqrt(M)), i);
    bar(V_sorted(:,i));
    title(['Principal Component ', num2str(i)]);
end

% Initialize the reconstructed image
reconstructed_img = zeros(H,W);

% Perform PCA on each 8x8 block
for i = 1:block_size:H
    for j = 1:block_size:W
        % Extract the 8x8 block
        block = img(i:i+block_size-1, j:j+block_size-1);

        % Vectorize the block to a 64-element vector
        block_vector = block(:);

        % Reshape block_vector to an 8x8 matrix
        block_matrix = reshape(block_vector, block_size, block_size);

        % Perform PCA on each row
        block_pca = zeros(block_size, M);
        for row = 1:block_size
            % Project each row of the block matrix onto the K principal components
            block_pca(row, :) = V_sorted(:, 1:M)' * block_matrix(row, :)';
        end

        % Reconstruct the block from K principal components
        block_reconstructed_matrix = zeros(block_size, block_size);
        for row = 1:block_size
            block_reconstructed_matrix(row, :) = (V_sorted(:, 1:M) * block_pca(row, :)')';
        end

        % Flatten the block_reconstructed_matrix back to a 64-element vector
        block_reconstructed_vector = block_reconstructed_matrix(:);

        % Reshape to 8x8 block
        block_reconstructed = reshape(block_reconstructed_vector, block_size, block_size);

        % Insert the reconstructed block back into the reconstructed image
        reconstructed_img(i:i+block_size-1, j:j+block_size-1) = block_reconstructed;
    end
end


% Display the original image
figure;
subplot(2,2,1);
imshow(img, []);
title('Original Image');

% Display the reconstructed image
subplot(2,2,2);
imshow(reconstructed_img, []);
title('Reconstructed Image');

% Compute and display the absolute error
abs_error = abs(img - reconstructed_img);
mean_error = round(mean(abs_error(:)), 2);
subplot(2,2,3);
imshow(abs_error, []); title(['Absolute Error Plot (MAE: ', num2str(mean_error), ')']);

% Define the maximum number of principal components to test
max_components = 8;

% Initialize an array to store MAE for each number of principal components
mae_values = zeros(1, max_components);

% Loop over the number of principal components to use for reconstruction
for I = 1:max_components
    % Perform 1D PCA on each 8x8 block (row-wise)
    for i = 1:block_size:H
        for j = 1:block_size:W
            % Extract the 8x8 block
            block = img(i:i+block_size-1, j:j+block_size-1);

            % Vectorize the block to a 64-element vector
            block_vector = block(:);

            % Reshape block_vector to an 8x8 matrix
            block_matrix = reshape(block_vector, block_size, block_size);

            % Perform PCA on each row
            block_pca = zeros(block_size, I);
            for row = 1:block_size
                % Project each row of the block matrix onto the K principal components
                block_pca(row, :) = V_sorted(:, 1:I)' * block_matrix(row, :)';
            end

            % Reconstruct the block from K principal components
            block_reconstructed_matrix = zeros(block_size, block_size);
            for row = 1:block_size
                block_reconstructed_matrix(row, :) = (V_sorted(:, 1:I) * block_pca(row, :)')';
            end

            % Flatten the block_reconstructed_matrix back to a 64-element vector
            block_reconstructed_vector = block_reconstructed_matrix(:);

            % Reshape to 8x8 block
            block_reconstructed = reshape(block_reconstructed_vector, block_size, block_size);

            % Insert the reconstructed block back into the reconstructed image
            reconstructed_img(i:i+block_size-1, j:j+block_size-1) = block_reconstructed;
        end
    end
    % Compute the mean absolute error for this number of principal components
    mae_values(I) = mean(abs(img(:) - reconstructed_img(:)));
end

% Plot MAE vs. number of principal components
figure;
plot(1:max_components, mae_values, 'bo-', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Mean Absolute Error');
title('MAE vs. Number of Principal Components');
grid on;


% c)

% number of PCAs to keep (equals to the previously declared M)
N = 16;
block_size = 8;
[H,W] = size(img);

num_blocks = ceil(H / block_size) * ceil(W / block_size);
X = zeros(block_size^2, num_blocks);
fprintf("Size of X: %d %d\n", size(X));

% Fill the X matrix
block_idx = 1;
for i = 1:block_size:H-block_size + 1
    for j = 1:block_size:W-block_size + 1
        block = img(i:i+block_size-1, j:j+block_size-1);
        X(:, block_idx) = block(:);
        block_idx = block_idx + 1;
    end
end

% Construct the covariance matrix
C = X * X';
fprintf("Size of C: %d %d\n", size(C));
[V, D] = eig(C);

% Extract and sort eigenvalues in descending order
eigenvalues = diag(D);
[eigenvalues_sorted, idx] = sort(eigenvalues, 'descend');

% Sort eigenvectors based on the sorted eigenvalues
V_sorted = V(:, idx);

% Plot the eigenvalues in descending order
figure;
subplot(1,2,1);
plot(log10(eigenvalues_sorted), 'bo-', 'LineWidth', 2);
title('Sorted Eigenvalues (Log Scale)');
xlabel('Eigenvalue Index');
ylabel('Eigenvalue');

subplot(1,2,2);
orthornormality_check = V' * V;
imagesc(orthornormality_check);
colormap(gray);
title("Orthonormality Check");

% Define the maximum number of principal components to test
max_components = 32;

% Initialize an array to store MAE for each number of principal components
mae_values = zeros(1, max_components);

% Loop over the number of principal components to use for reconstruction
for I = 1:max_components
    % Initialize the reconstructed data matrix
    X_reconstructed = zeros(size(X));
    
    % Project the data onto the K principal components and reconstruct
    for k = 1:size(X, 2)
        X_reconstructed(:, k) = V_sorted(:, 1:I) * (V_sorted(:, 1:I)' * X(:, k));
    end
    
    % Reconstruct the image from the data matrix
    reconstructed_img = zeros(H, W);
    block_index = 1;
    for i = 1:block_size:H-block_size+1
        for j = 1:block_size:W-block_size+1
            block_reconstructed = reshape(X_reconstructed(:, block_index), block_size, block_size);
            reconstructed_img(i:i+block_size-1, j:j+block_size-1) = block_reconstructed;
            block_index = block_index + 1;
        end
    end
    
    % Compute the mean absolute error for this number of principal components
    mae_values(I) = mean(abs(img(:) - reconstructed_img(:)));
end

% Plot MAE vs. number of principal components
figure;
plot(1:max_components, mae_values, 'bo-', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Mean Absolute Error');
title('MAE vs. Number of Principal Components');
grid on;

% Reconstruct the image
X_reconstructed = zeros(size(X));
for k = 1:size(X, 2)
    X_reconstructed(:, k) = V_sorted(:, 1:N) * (V_sorted(:, 1:N)' * X(:, k));
end

% Reconstruct the image from the data matrix
reconstructed_img = zeros(H, W);
block_index = 1;
for i = 1:block_size:H-block_size+1
    for j = 1:block_size:W-block_size+1
        block_reconstructed = reshape(X_reconstructed(:, block_index), block_size, block_size);
        reconstructed_img(i:i+block_size-1, j:j+block_size-1) = block_reconstructed;
        block_index = block_index + 1;
    end
end

% Display the original image
figure;
subplot(2,2,1);
imshow(img2, []);
title('Original Image');

% Display the reconstructed image
subplot(2,2,2);
imshow(reconstructed_img, []);
title('Reconstructed Image');

% Compute and display the absolute error
abs_error = abs(img - reconstructed_img);
mean_error = round(mean(abs_error(:)), 2);
subplot(2,2,3);
imshow(abs_error, []); title(['Absolute Error Plot (MAE: ', num2str(mean_error), ')']);


% Display the 2D principal components
figure;
for i = 1:N
    subplot(ceil(sqrt(N)), ceil(sqrt(N)), i);
    imagesc(reshape(V_sorted(:, i), block_size, block_size));
    colormap(gray);
    title(['Principal Component ', num2str(i)]);
    axis square;
end