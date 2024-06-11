
% ----------------- Problem 3: Restoration ----------------

% 3.1

% 2D Gaussian kernel
h = fspecial('gaussian', 7, 1);
figure;
image(h/max(h(:))*255);
colormap gray;
title('2D Gaussian Kernel');

% Kernel details
[K, ~] = size(h);
K = floor(K/2);

% H matrix size
M = 256; 
N = 256;
num_pixels = M * N;

% Efficient construction of the sparse blur matrix H

offsets = -K:K;
num_offsets = numel(offsets);
h_flattened = h(:);
total_entries = num_offsets^2 * num_pixels;

I = zeros(total_entries, 1);
J = zeros(total_entries, 1);
V = zeros(total_entries, 1);

index = 1;

for i = 1:M
    for j = 1:N
        center_idx = (i - 1) * N + j;
        for m = 1:num_offsets
            for n = 1:num_offsets
                row_offset = offsets(m);
                col_offset = offsets(n);
                
                row_idx = mod(i - 1 + row_offset + M, M) + 1;
                col_idx = mod(j - 1 + col_offset + N, N) + 1;
                neighbor_idx = (row_idx - 1) * N + col_idx;
                
                I(index) = center_idx;
                J(index) = neighbor_idx;
                V(index) = h_flattened((m - 1) * num_offsets + n);
                index = index + 1;
            end
        end
    end
end

H = sparse(I, J, V);

% Display sparse matrix
figure;
spy(H);
title('Sparse Matrix H');

% 3.2

% Load and display the image
a = imread('barbara.png');
a = imresize(a, [N,M]);
a = im2double(a);

figure;
subplot(2,5,1);
imshow(a, []);
title('Original Image');

% Apply the blur by b = H*a
a_vec = a(:); % Vectorize the image
b = H * a_vec;
b_img = reshape(b, M, N); % Reshape to 2D image

subplot(2,5,2);
imshow(b_img, []);
title('Blurred Image');

% 3.3

% Add observation noise
b_noisy = b + randn(size(b)) * 0.01;
b_noisy_img = reshape(b_noisy, M, N);

subplot(2,5,3);
imshow(b_noisy_img, []);
title('Blurred Image with Noise');

% Restore the image by solving the minimization problem
lambda_vals = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
errors = zeros(length(lambda_vals), 1);
norms = zeros(length(lambda_vals), 1);

for k = 1:length(lambda_vals)
    lambda = lambda_vals(k);
    A = H' * H + lambda * speye(M*N);
    b_restored = A \ (H' * b_noisy);
    a_restored = reshape(b_restored, M, N);

    % Plot and calculate ||b - Ha||^2 and λ||a||^2
    Ha = H * b_restored;
    errors(k) = norm(b_noisy - Ha)^2;
    norms(k) = norm(b_restored)^2;
    
    subplot(2,5,k+3);
    imshow(a_restored, []);
    title(['Restored Image, \lambda = ', num2str(lambda)]);
end

% Plot ||b - Ha||^2 vs λ||a||^2
figure;
loglog(norms, errors, '-o');
xlabel('\lambda ||a||^2');
ylabel('||b - Ha||^2');
title('Trade-off curve for different \lambda values');
grid on;

% 3.4

% Restore using gradient descent
lambda_best = 1e-2;
[a_restored_grad_descent, cost_descent] = gradient_descent(a, H, b_noisy, lambda_best, M, N);

figure;
subplot(1,3,1);
imshow(a_restored_grad_descent, []);
title(['Restored Image with Gradient Descent, \lambda = ', num2str(lambda_best)]);

% Regularization matrix
I = speye(M*N);
Q = H' * H + lambda * I;
b_eff = H' * b_noisy;

[a_restored_steep_descent, cost_steep] = steepest_descent(a, H, lambda_best, Q, b_eff, M, N);

subplot(1,3,2);
imshow(a_restored_steep_descent, []);
title(['Restored Image with Steepest Descent, \lambda = ', num2str(lambda_best)]);

[a_restored_conj, cost_conj] = conjugate_gradient(a, H, lambda_best, Q, b_eff, M, N);

subplot(1,3,3);
imshow(a_restored_conj, []);
title(['Restored Image with Conjugate Gradient, \lambda = ', num2str(lambda_best)]);

% Plot the cost vs. iteration number
figure;
plot(cost_descent);
hold on;
plot(cost_steep);
hold on;
plot(cost_conj);
xlabel('Iteration Number');
ylabel('Cost Function Value');
legend('Gradient Descent', 'Steepest Descent', 'Conjugate Gradient');
title('Cost Function vs. Iteration Number');
grid on;


% Gradient descent function
function [a_restored, cost_history] = gradient_descent(a, H, b, lambda, M, N)
    max_iter = 100;
    tol = 1e-6;
    alpha = 1e-3;

    a_restored = zeros(M*N, 1);
    cost_history = zeros(max_iter, 1);
    for iter = 1:max_iter

        cost_history(iter) = norm(a(:) - H * a_restored)^2 + lambda * norm(a_restored)^2; % Cost function
        grad = H' * (H * a_restored - b) + lambda * a_restored;
        a_restored = a_restored - alpha * grad;
        
        if norm(grad) < tol
            cost_history = cost_history(1:iter); % Trim unused entries
            break;
        end
    end

    a_restored = reshape(a_restored, M, N);
end

% Steepest Descent Algorithm
function [x_img, cost_history] = steepest_descent(a, H, lambda, Q, b, M, N)
    x = zeros(M*N, 1);
    max_iter = 100;
    tol = 1e-6;
    alpha = 1e-3;
    cost_history = zeros(max_iter, 1);
    for iter = 1:max_iter
        g = Q * x - b; % Gradient
        cost_history(iter) = norm(a(:) - H * x)^2 + lambda * norm(x)^2; % Cost function
        
        if norm(g) < tol
            cost_history = cost_history(1:iter); % Trim unused entries
            break;
        end
        
        alpha = (g' * g) / (g' * Q * g); % Step size
        x = x - alpha * g; % Descent step
    end
    x_img = reshape(x, M, N);
end

% Conjugate Gradient Algorithm
function [x_img, cost_history] = conjugate_gradient(a, H, lambda, Q, b, M, N)
    x = zeros(M*N,1);
    max_iter = 100;
    tol = 1e-6;
    alpha = 1e-3;
    g = Q * x - b; % Gradient
    d = -g; % Initial direction
    cost_history = zeros(max_iter, 1);
    for iter = 1:max_iter
        cost_history(iter) = norm(a(:) - H * x)^2 + lambda * norm(x)^2; % Cost function
        if norm(g) < tol
            cost_history = cost_history(1:iter); % Trim unused entries
            break;
        end
        alpha = (g' * g) / (d' * Q * d); % Step size
        x = x + alpha * d; % Descent step
        g_new = Q * x - b; % New gradient
        beta = (g_new' * Q * d) / (d' * Q * d); % Weight
        d = -g_new + beta * d; % Update direction
        g = g_new; % Update gradient
    end
    x_img = reshape(x, M, N);
end