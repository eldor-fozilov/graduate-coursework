
% ----------------- Problem 2: Detection ----------------

rand_image = imread("face/1.pgm");
[H,W] = size(rand_image);

N = 20;
L = zeros(H*W, N);
fprintf("Size of L: %d %d\n", size(L));

% 2.1

% load the data
for i=1:N
    img = double(imread(sprintf("face/%d.pgm", i)));
    subplot(4, 5, i); imshow(uint8(img)); title(['Image ', num2str(i)]);
    L(:,i) = img(:);
end

% Calculate the L mean
L_mean = mean(L, 2);
% Plot
figure;
imagesc(uint8(reshape(L_mean, [H, W])));
colormap gray;
axis off;
title("Mean of the Images");

L_centered = L - L_mean;
C = (L_centered * L_centered') / size(L_centered, 1);
[V,D] = eig(C);

% Extract and sort eigenvalues in descending order
eigenvalues = diag(D);
[eigenvalues_sorted, idx] = sort(eigenvalues, 'descend');

% Sort eigenvectors based on the sorted eigenvalues
V_sorted = V(:, idx);

figure;
for i = 1:3
    subplot(1, 3, i); imagesc(reshape(V_sorted(:, i), [H, W]));
    colormap gray;
    axis off;
    title(['Eigenface ', num2str(i)]);
end

% 2.2

T = V(:, 1:3)';
x = T * L_centered;
y = [ones(10, 1); -ones(10, 1)]; % first 10 male faces and the last 10 female faces

figure;
scatter3(x(1, :), x(2, :), x(3, :));
title("Scatter plot of the compressed data.");

% Sample training data (indices)
random_indices = randperm(10);
random_indices = [random_indices(1:8), random_indices(1:8) + 10];

% Initialize the learnable parameters
w = zeros(size(x, 1), 1);
b = 0;

% Hyperparameters
iters = 50;
step_size = 0.005;
lambda = 1 / iters;

% 2.3

% TRAIN
for iter = 1:iters
    for i = 1:16
        random_idx = random_indices(i);
        x_sample = x(:, random_idx);
        y_sample = y(random_idx);
        pred = sign(w' * x_sample + b);

        if pred * y_sample < 1
            dw = -y_sample * x_sample;
            db = -y_sample;
        else
            dw = 0;
            db = 0;
        end
        w = w - step_size * (dw + lambda * w);
        b = b - step_size * db;
    end
end

% Get the indices for the test dataset
test_indices = setdiff(1:20, random_indices);

% 2.4

% TEST
test_data = x(:, test_indices);
test_labels = y(test_indices);

predictions = sign(w' * test_data + b);
predictions = predictions(:);
display(test_labels);
display(predictions);
labels = test_labels;

true_positive = sum((predictions == 1) & (labels == 1));
false_negative = sum((predictions == -1) & (labels == 1));
false_positive = sum((predictions == 1) & (test_labels == -1));
true_negative = sum((predictions == -1) & (test_labels == -1));

precision = true_positive / (true_positive + false_positive);
recall = true_positive / (true_positive + false_negative);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);