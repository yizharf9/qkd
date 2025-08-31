
function final_signal = generate_gamma_gamma_signal(N, tau_c, alpha, beta)
    % Step 1: Generate AR(1) Gaussian process
    a = exp(-1 / tau_c);
    w = randn(N, 1);
    gaussian_seq = filter(1, [1 -a], w);

    % Step 2: Generate Gamma-Gamma sequence
    gamma1 = gamrnd(alpha, 1/alpha, N, 1);
    gamma2 = gamrnd(beta, 1/beta, N, 1);
    gamma_gamma_seq = gamma1 .* gamma2;

    % Step 3: Rank Matching
    [~, sort_idx] = sort(gaussian_seq);
    [~, inv_idx] = sort(sort_idx);
    sorted_gamma = sort(gamma_gamma_seq);
    final_signal = sorted_gamma(inv_idx);
end

function acf = empirical_autocorrelation(x, max_lag)
    x = x - mean(x);
    N = length(x);
    acf = zeros(max_lag+1, 1);
    for k = 0:max_lag
        acf(k+1) = sum(x(1:N-k) .* x(k+1:N)) / (N - k);
    end
    acf = acf / acf(1);  % Normalize
end
function [alpha, beta, sigma_R2] = compute_alpha_beta(L, lambda_nm, Cn2)
    % Inputs:
    % L - path length (meters)
    % lambda_nm - wavelength (nanometers)
    % Cn2 - refractive index structure parameter (m^(-2/3))

    % Convert wavelength to meters
    lambda = lambda_nm * 1e-9;
    k = 2 * pi / lambda;

    % Compute Rytov variance
    sigma_R2 = 1.23 * Cn2 * k^(7/6) * L^(11/6);

    % Compute alpha
    num_alpha = 0.49 * sigma_R2;
    den_alpha = (1 + 1.11 * sigma_R2^(12/5))^(7/6);
    alpha = 1 / (exp(num_alpha / den_alpha) - 1);

    % Compute beta
    num_beta = 0.51 * sigma_R2;
    den_beta = (1 + 0.69 * sigma_R2^(12/5))^(5/6);
    beta = 1 / (exp(num_beta / den_beta) - 1);
end

% Demo Script
N = 10000;
tau_c = 20;
L = 200e3; %set the length of channel to 20km
lambda_t_nm = 1550;
sigma_n = 7.4890e-17;
[alpha,beta,sigma_R2]=compute_alpha_beta(L,lambda_t_nm,sigma_n)

signal = generate_gamma_gamma_signal(N, tau_c, alpha, beta);

% Plot ACF
max_lag = 100;
emp_acf = empirical_autocorrelation(signal, max_lag);
theo_acf = exp(-[0:max_lag]' / tau_c);

figure;
plot(0:max_lag, emp_acf, 'b', 'DisplayName', 'Empirical ACF');
hold on;
plot(0:max_lag, theo_acf, 'r--', 'DisplayName', 'Theoretical ACF');
xlabel('Lag');
ylabel('Autocorrelation');
legend;
title('Autocorrelation Comparison');
grid on;

% Plot PDF
figure;
histogram(signal, 100, 'Normalization', 'pdf');
title('Gamma-Gamma Signal PDF');
xlabel('Value');
ylabel('Density');
grid on;
