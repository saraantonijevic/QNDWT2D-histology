clear
close all force
clc

%check dependency
if exist('WavmatQBlock','file') ~= 2
    error('WavmatQBlock.m is not on the MATLAB path.')
end

%Donoho-Johnstone Blocks signal
N = 1024;
t = ((1:N)' - 0.5) / N;

pos = [0.10 0.13 0.15 0.23 0.25 0.40 0.44 0.65 0.76 0.78 0.81];
hgt = [4.00 -5.00 3.00 -4.00 5.00 -4.20 2.10 4.30 -3.10 2.10 -4.20];

x = zeros(N,1);
for k = 1:length(pos)
    x = x + hgt(k) * (1 + sign(t - pos(k))) / 2;
end

%adding Gaussian noise with SNR = Var(signal)/Var(noise)
% force sample Var(noise) = 1 exactly
SNR_target = 7;% ratio, not dB
noiseVarTarget = 1; % exact sample variance in generated noise

vx = var(x,1);
xtrue = sqrt(SNR_target * noiseVarTarget / vx) * x;

rng(0)
noise = randn(N,1);
noise = noise - mean(noise);
noise = sqrt(noiseVarTarget) * noise / std(noise,1); % exact var(noise,1)=1

xnoisy = xtrue + noise;

%quaternion valued filters and transform matrix
levels = 5;
shift=2;
side = 'L';

[h, g] = qginzberg10();
W = WavmatQBlock(h, g, N, levels, shift, side);

%prepare a fast inverse operator
M = size(W,1);
if issparse(W)
    Iden = speye(M);
else
    Iden = eye(M);
end

orthErr = norm(W' * W - Iden, 'fro') / sqrt(M);

if orthErr < 1e-10
    useTransposeInverse = true;
    Winv_apply = @(y) W' * y;
else
    useTransposeInverse = false;
    if exist('decomposition','builtin') || exist('decomposition','file')
        Wfac = decomposition(W,'lu');
        Winv_apply = @(y) Wfac \ y;
    else
        Winv_apply = @(y) W \ y;
    end
end

% embed real signal as scalar quaternion signal
xq = zeros(4*N,1);
xq(1:4:end) = xnoisy;

% forward quaternion wavelet transform
wq = W * xq;
Wcoef = reshape(wq, 4, []).';

%identify coefficient blocks
[idxA, idxD] = qwt_block_indices(N, levels);

%two-parameter oracle tuning
%    lambda_j = base * 2^(-c*(j-1)) * sqrt(2*log(n_j))
targetMSE = 0.09;

% Coarse search
baseGrid1 = 0.20:0.02:1.20;
cGrid1 = 0.00:0.05:2.00;

[bestBase1, bestC1, bestMSE1, bestWcoef1] = ...
    oracle_search_bc(Wcoef, idxD, levels, N, Winv_apply, xtrue, baseGrid1, cGrid1);

% Refined search
baseMin2 = max(0.05, bestBase1 - 0.08);
baseMax2 = bestBase1 + 0.08;
cMin2= max(0.00, bestC1 - 0.25);
cMax= bestC1 + 0.25;

baseGrid2 = baseMin2:0.005:baseMax2;
cGrid2 = cMin2:0.02:cMax2;

[bestBase, bestC, bestMSE, bestWcoef] = ...
    oracle_search_bc(Wcoef, idxD, levels, N, Winv_apply, xtrue, baseGrid2, cGrid2);

Wcoef_thr = bestWcoef;

% winning lambdas by level
lambda_best = zeros(levels,1);
alpha_best  = zeros(levels,1);

for lev = 1:levels
    nj = numel(idxD{lev});
    alpha_best(lev)  = 2^(-bestC*(lev-1));
    lambda_best(lev) = bestBase * alpha_best(lev) * sqrt(2*log(nj));
end

%inverse transform with winning coefficients
wq_thr = reshape(Wcoef_thr.', 4*N, 1);
xq_rec = Winv_apply(wq_thr);
xrec= xq_rec(1:4:end);

%Diagnostics
mse_noisy = mean((xtrue - xnoisy).^2);
mse_rec= mean((xtrue - xrec).^2);

noise_in= xnoisy - xtrue;
noise_out = xrec - xtrue;

snr_in_ratio = var(xtrue,1) / var(noise_in,1);
snr_out_ratio = var(xtrue,1) / var(noise_out,1);

snr_in_dB  = 10*log10(snr_in_ratio);
snr_out_dB = 10*log10(snr_out_ratio);

xq_exact = Winv_apply(wq);
x_exact = xq_exact(1:4:end);
exactInvErr = max(abs(x_exact - xnoisy));
%summary prints
fprintf('\n');
fprintf('Quaternion wavelet denoising with WavmatQBlock\n');
fprintf('Blocks signal, Donoho-Johnstone test function\n');
fprintf('N = %d\n', N);
fprintf('Levels  = %d\n', levels);
fprintf('Shift= %d\n', shift);
fprintf('Side  = %s\n', side);
fprintf('Orthogonality error= %.3e\n', orthErr);
fprintf('Use W'' as inverse = %d\n', useTransposeInverse);
fprintf('Exact inv max error = %.3e\n', exactInvErr);


fprintf('Signal/noise setup\n');
fprintf('Target SNR ratio = %.6f\n', SNR_target);
fprintf('Var(xtrue)= %.6f\n', var(xtrue,1));
fprintf('Var(noise) = %.6f\n', var(noise,1));
fprintf('Actual input ratio = %.6f\n', snr_in_ratio);
fprintf('Actual input dB  = %.6f dB\n', snr_in_dB);
fprintf('\n');

fprintf('Oracle search results\n');
fprintf('Coarse best base = %.6f\n', bestBase1);
fprintf('Coarse best c= %.6f\n', bestC1);
fprintf('Coarse best MSE  = %.6f\n', bestMSE1);
fprintf('\n');
fprintf('Refined best base= %.6f\n', bestBase);
fprintf('Refined best c = %.6f\n', bestC);
fprintf('Refined best MSE  = %.6f\n', bestMSE);
fprintf('\n');

fprintf('Final denoising results\n');
fprintf('Noisy MSE = %.6f\n', mse_noisy);
fprintf('Recon MSE  = %.6f\n', mse_rec);
fprintf('Output SNR ratio = %.6f\n', snr_out_ratio);
fprintf('Output SNR dB = %.6f dB\n', snr_out_dB);
fprintf('\n');

fprintf('Winning lambdas by detail level\n');
for lev = 1:levels
    nj = numel(idxD{lev});
    fprintf('D%d: n_j = %4d, alpha_j = %.6f, lambda_j = %.6f\n', ...
        lev, nj, alpha_best(lev), lambda_best(lev));
end
fprintf('\n');

if mse_rec < targetMSE
    fprintf('Target achieved: Recon MSE = %.6f < %.6f\n', mse_rec, targetMSE);
else
    fprintf('Target not achieved: Recon MSE = %.6f >= %.6f\n', mse_rec, targetMSE);
    fprintf('Try widening the search box, increasing levels, or switching filter family.\n');
end

% plots
figure('Color','w');

subplot(4,1,1)
plot(t, xtrue, 'k', 'LineWidth', 1.5); hold on
plot(t, xnoisy, 'r')
grid on
title('Original and noisy Blocks signal')
legend('Original','Noisy','Location','best')

subplot(4,1,2)
plot(t, xrec, 'b', 'LineWidth', 1.5); hold on
plot(t, xtrue, 'k--', 'LineWidth', 1.0)
grid on
title(sprintf('Quaternion wavelet denoised reconstruction, MSE = %.5f', mse_rec))
legend('Denoised','Original','Location','best')

subplot(4,1,3)
plot(t, xtrue - xrec, 'm')
grid on
title('Reconstruction error')

subplot(4,1,4)
plot_quaternion_coeff_norms(Wcoef, idxA, idxD, levels)
title('Quaternion coefficient norms by block')
grid on

% local functions
function [bestBase, bestC, bestMSE, bestWcoef] = ...
    oracle_search_bc(Wcoef, idxD, levels, N, Winv_apply, xtrue, baseGrid, cGrid)

    bestMSE = inf;
    bestBase = NaN;
    bestC = NaN;
    bestWcoef = Wcoef;

    for base = baseGrid
        for c = cGrid

            Wtmp = apply_qsoft_threshold(Wcoef, idxD, levels, base, c);

            wq_tmp = reshape(Wtmp.', 4*N, 1);
            xq_tmp = Winv_apply(wq_tmp);
            xtmp = xq_tmp(1:4:end);

            mse_tmp = mean((xtrue - xtmp).^2);

            if mse_tmp < bestMSE
                bestMSE = mse_tmp;
                bestBase  = base;
                bestC  = c;
                bestWcoef = Wtmp;
            end
        end
    end
end

function Wcoef_thr = apply_qsoft_threshold(Wcoef, idxD, levels, base, c)
% Quaternion vector soft thresholding on detail blocks
% lambda_j = base * 2^(-c*(j-1)) * sqrt(2*log(n_j))

    Wcoef_thr = Wcoef;

    for lev = 1:levels
        idx = idxD{lev};
        qblock = Wcoef_thr(idx,:);
        mag = sqrt(sum(qblock.^2, 2));

        nj = numel(idx);
        alpha_j  = 2^(-c*(lev-1));
        lambda_j = base * alpha_j * sqrt(2*log(nj));

        shrink = max(0, 1 - lambda_j ./ max(mag, eps));
        Wcoef_thr(idx,:) = qblock .* repmat(shrink, 1, 4);
    end
end

function [idxA, idxD] = qwt_block_indices(N, levels)
% Ordering is [A_levels, D_levels, D_{levels-1}, ..., D_1]

    aLen = N / 2^levels;
    idxA = 1:aLen;

    idxD = cell(levels,1);
    pos = aLen + 1;

    for j = levels:-1:1
        len = N / 2^j;
        idxD{j} = pos:(pos + len - 1);
        pos = pos + len;
    end
end

function plot_quaternion_coeff_norms(Wcoef, idxA, idxD, levels)
    qn = sqrt(sum(Wcoef.^2, 2));
    plot(qn, 'k', 'LineWidth', 1.0); hold on

    xline(idxA(end) + 0.5, 'b-');

    for j = levels:-1:1
        xline(idxD{j}(end) + 0.5, 'r-');
    end

    xlabel('Coefficient index')
    ylabel('Quaternion norm')
end

function [h, g] = qginzberg10()
% Ginzberg 10 tap quaternion filters each row [a b c d] means a + b*i + c*j + d*k

    C1 = sqrt(2)/256;
    C2 = sqrt(35)/256;
    C3 = 1/24576;
    C4 = 1/3072;
    C5 = 1/256;
    C6 = 1/12288;

    h = zeros(10,4);
    g = zeros(10,4);

    % low pass
    h(1,:)  = [0, C2,  0,  0];
    h(2,:)  = [-5*C1, 0,  0, C2];
    h(3,:)  = [-7*C1, -7*C2,  0, 3*C2];
    h(4,:) = [35*C1, -5*C2,  0, C2];
    h(5,:)  = [105*C1, 11*C2,  0, -5*C2];
    h(6,:) = h(5,:);
    h(7,:) = h(4,:);
    h(8,:) = h(3,:);
    h(9,:) = h(2,:);
    h(10,:) = h(1,:);

    % high pass
    g0 = [0, ...
          89*sqrt(35)*C3, ...
          35*sqrt(2)*C3, ...
         -35*sqrt(35)*C3];

    g1 = [-480*sqrt(2)*C3, ...
           35*sqrt(35)*C3, ...
         -175*sqrt(2)*C3, ...
           79*sqrt(35)*C3];

    g2 = [84*sqrt(2)*C4, ...
         -91*sqrt(35)*C4, ...
          35*sqrt(2)*C4, ...
           sqrt(35)*C4];

    g3 = [35*sqrt(2)*C5, ...
           5*sqrt(35)*C5, ...
           0, ...
          -sqrt(35)*C5];

    g4 = [-5040*sqrt(2)*C6, ...
            577*sqrt(35)*C6, ...
           -245*sqrt(2)*C6, ...
              5*sqrt(35)*C6];

    g(1,:) =  g0;
    g(2,:) =  g1;
    g(3,:) =  g2;
    g(4,:)  =  g3;
    g(5,:) =  g4;
    g(6,:) = -g4;
    g(7,:) = -g3;
    g(8,:)  = -g2;
    g(9,:) = -g1;
    g(10,:) = -g0;
end