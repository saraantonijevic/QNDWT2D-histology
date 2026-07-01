function [hier, info] = qndwt2d2(A, h, g, L, shape, varargin)
% This version uses the standard two-sided image-transform convention
%B = W_1 A W_2^*
%rather than the earlier qndwt2d convention B = W_1^* A W_2.
%
% Quaternion convention:
% quaternion image is stored as m1 by m2 by 4, with components
%      A(:,:,1) + A(:,:,2)*i + A(:,:,3)*j + A(:,:,4)*k.
%real or complex m1 by m2 input is embedded automatically as real + imag*i + 0*j + 0*k.
%
% A: real, complex, or quaternion image. Quaternion images are m1 by m2 by 4 arrays
% h :low-pass quaternion analysis filter, size K by 4
% g :high-pass quaternion analysis filter, size K by 4. Each row is [a b c d] = a + b*i + c*j + d*k.
% L : number of nondecimated levels. shape   'd', 'h', 'v', or 'all'.
%
% Output:
%  hier:If shape is 'd', 'h', or 'v', hier is a quaternion stack of size (L+1)*m1 by m2 by 4. The smooth block S_L is included.
%           If shape is 'all', hier is a structure with fields d, h, v
%
% Shift: integer circular shift. Default 0.
% DetailOrder: 'fine-to-coarse' or 'coarse-to-fine'
%    Default 'fine-to-coarse'
% SmoothPosition: 'first' or 'last'. Default 'first'
%
% At recursive level lev, the dilated filters define low-pass and high-pass nondecimated convolution operators H_lev and G_lev.
% The same-scale quaternion blocks follow the standard matrix convention
%S_lev = H_lev A_{lev-1} H_lev^*,
  %H_lev = G_lev A_{lev-1} H_lev^*,
% V_lev = H_lev A_{lev-1} G_lev^*,
%  D_lev = G_lev A_{lev-1} G_lev^*.
%
%   Here H_lev as a detail block is stored in the field/shape 'h', while  the low-pass filtering operator is described by the filter h. In the
%   code, Hcell denotes horizontal details and hlev denotes the low-pass filter at level lev.
%
%   Operationally, left filtering uses left quaternion multiplication by filter taps, and right adjoint filtering uses right multiplication by
%   conjugated filter taps. The large convolution matrices are not formed; the action is implemented using circular shifts.


if nargin < 5
    error('Usage: hier = qndwt2d2(A,h,g,L,shape,...).');
end

if ~isscalar(L) || L < 1 || L ~= round(L)
    error('L must be a positive integer.');
end

if isempty(h) || isempty(g) || size(h,2) ~= 4 || size(g,2) ~= 4
    error('h and g must be nonempty K by 4 quaternion filter arrays.');
end

p = inputParser;
p.addParameter('Shift', 0, @(x) isnumeric(x) && isscalar(x));
p.addParameter('DetailOrder', 'fine-to-coarse', @(x) ischar(x) || isstring(x));
p.addParameter('SmoothPosition', 'first', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opt = p.Results;

shape = lower(string(shape));
detailOrder = lower(string(opt.DetailOrder));
smoothPosition = lower(string(opt.SmoothPosition));

if ~ismember(shape, ["d","h","v","all"])
    error("shape must be 'd', 'h', 'v', or 'all'.");
end
if ~ismember(detailOrder, ["fine-to-coarse","coarse-to-fine"])
    error("DetailOrder must be 'fine-to-coarse' or 'coarse-to-fine'.");
end
if ~ismember(smoothPosition, ["first","last"])
    error("SmoothPosition must be 'first' or 'last'.");
end

Acur = qimage_local(A);
hf = double(h);
gf = double(g);

[m1,m2,~] = size(Acur);

Dcell = cell(L,1);
Hcell = cell(L,1);
Vcell = cell(L,1);

energyD = nan(L,1);
energyH = nan(L,1);
energyV = nan(L,1);
energyS = nan(L,1);

for lev = 1:L

    gap = 2^(lev-1) - 1;

    hlev = dilate_qfilter_local(hf, gap);
    glev = dilate_qfilter_local(gf, gap);

    if size(hlev,1) > m1 || size(glev,1) > m1 || ...
       size(hlev,1) > m2 || size(glev,1) > m2
        error('At level %d the dilated filter is longer than at least one image dimension.', lev);
    end

    % Standard convention B = W_1 A W_2^*
    % First filter rows from the left, then filter columns from the right by the adjoint/conjugate filter
    HA = apply_left_q_local(Acur, hlev, opt.Shift);
    GA = apply_left_q_local(Acur, glev, opt.Shift);

    Sblock = apply_right_adjoint_q_local(HA, hlev, opt.Shift);  % H A H^*
    Vblock= apply_right_adjoint_q_local(HA, glev, opt.Shift);  % H A G^*
    Hblock = apply_right_adjoint_q_local(GA, hlev, opt.Shift);  % G A H^*
    Dblock = apply_right_adjoint_q_local(GA, glev, opt.Shift);  % G A G^*

    Hcell{lev} = Hblock;
    Vcell{lev} = Vblock;
    Dcell{lev} = Dblock;

    energyH(lev) = qenergy_local(Hblock);
    energyV(lev) = qenergy_local(Vblock);
    energyD(lev) = qenergy_local(Dblock);
    energyS(lev) = qenergy_local(Sblock);

    Acur = Sblock;
end

smooth = Acur;
smoothEnergy = qenergy_local(smooth);

switch shape
    case "d"
        [hier, labels] = stack_qblocks_local(smooth, Dcell, 'D', detailOrder, smoothPosition);
    case "h"
        [hier, labels] = stack_qblocks_local(smooth, Hcell, 'H', detailOrder, smoothPosition);
    case "v"
        [hier, labels] = stack_qblocks_local(smooth, Vcell, 'V', detailOrder, smoothPosition);
    case "all"
        [hier.d, labelsD] = stack_qblocks_local(smooth, Dcell, 'D', detailOrder, smoothPosition);
        [hier.h, labelsH] = stack_qblocks_local(smooth, Hcell, 'H', detailOrder, smoothPosition);
        [hier.v, labelsV] = stack_qblocks_local(smooth, Vcell, 'V', detailOrder, smoothPosition);
        labels = struct('d', labelsD, 'h', labelsH, 'v', labelsV);
end

info = struct;
info.version = 'qndwt2d2_standard_B_equals_WAWstar_2026';
info.convention = 'B = W1 A W2^*';
info.shape = char(shape);
info.sizeA = [m1 m2];
info.L = L;
info.filterLowpass = hf;
info.filterHighpass = gf;
info.shift = opt.Shift;
info.detailOrder = char(detailOrder);
info.smoothPosition = char(smoothPosition);
info.blockLabels = labels;

info.energy.diagonal = energyD;
info.energy.horizontal = energyH;
info.energy.vertical = energyV;
info.energy.smoothByLevel = energyS;
info.energy.smooth = smoothEnergy;

info.log2EnergyDiagonal = log2(max(energyD, realmin));
info.log2EnergyHorizontal = log2(max(energyH, realmin));
info.log2EnergyVertical = log2(max(energyV, realmin));
info.log2EnergySmooth = log2(max(smoothEnergy, realmin));

switch shape
    case "d"
        info.energySelected = energyD;
        info.log2Energy = log2(max(energyD, realmin));
    case "h"
        info.energySelected = energyH;
        info.log2Energy = log2(max(energyH, realmin));
    case "v"
        info.energySelected = energyV;
        info.log2Energy = log2(max(energyV, realmin));
    case "all"
        info.energySelected = struct('d', energyD, 'h', energyH, 'v', energyV);
        info.log2Energy = struct('d', log2(max(energyD, realmin)), ...
                                 'h', log2(max(energyH, realmin)), ...
                                 'v', log2(max(energyV, realmin)));
end

end

%local quaternion helper functions

function Q = qimage_local(A)
% Convert real or complex image to quaternion image. If already quaternion,
% check that it is m by n by 4

if ~isnumeric(A)
    error('A must be numeric.');
end

if ismatrix(A)
    [m1,m2] = size(A);
    Q = zeros(m1,m2,4);
    Q(:,:,1) = real(double(A));
    Q(:,:,2) = imag(double(A));
    return;
end

if ndims(A) == 3 && size(A,3) == 4
    Q = double(A);
else
    error('Quaternion image A must be m by n by 4.');
end
end

function Y = apply_left_q_local(M, f, shift)
% Computes F * M; this is the row-side part of B = W A W^*

Y = zeros(size(M));
K = size(f,1);

for r = 0:(K-1)
    q = f(r+1,:);
    if any(q ~= 0)
        Ms = circshift(M, [(shift+r), 0, 0]);
        Y = Y + qmul_left_scalar_local(q, Ms);
    end
end
end

function Y = apply_right_adjoint_q_local(M, f, shift)
% Computes M * F^*. Each conjugated filter tap multiplies pixels on the right; this is the column-side adjoint part of B = W A W^*

Y = zeros(size(M));
K = size(f,1);

for r = 0:(K-1)
    q = qconj_scalar_local(f(r+1,:));
    if any(q ~= 0)
        Ms = circshift(M, [0, (shift+r), 0]);
        Y = Y + qmul_right_scalar_local(Ms, q);
    end
end
end

function filtd = dilate_qfilter_local(filt, gap)

if gap < 1
    filtd = filt;
    return;
end

K = size(filt,1);
newlength = (gap+1)*K - gap;
filtd = zeros(newlength,4);
filtd(1:(gap+1):newlength,:) = filt;
end

function [H, labels] = stack_qblocks_local(smooth, details, prefix, detailOrder, smoothPosition)

L = length(details);
[m1,m2,~] = size(smooth);

switch detailOrder
    case "fine-to-coarse"
        idx = 1:L;
    case "coarse-to-fine"
        idx = L:-1:1;
end

blocks = cell(L+1,1);
labels = strings(L+1,1);

if smoothPosition == "first"
    blocks{1} = smooth;
    labels(1) = "S_" + string(L);

    for ii = 1:L
        lev = idx(ii);
        blocks{ii+1} = details{lev};
        labels(ii+1) = string(prefix) + "_" + string(lev);
    end
else
    for ii = 1:L
        lev = idx(ii);
        blocks{ii} = details{lev};
        labels(ii) = string(prefix) + "_" + string(lev);
    end

    blocks{L+1} = smooth;
    labels(L+1) = "S_" + string(L);
end

H = zeros((L+1)*m1, m2, 4);

for b = 1:(L+1)
    rows = (b-1)*m1 + (1:m1);
    H(rows,:,:) = blocks{b};
end
end

function E = qenergy_local(Q)
tmp = sum(Q.^2,3);
E = mean(tmp(:));
end

function qc = qconj_scalar_local(q)
qc = [q(1), -q(2), -q(3), -q(4)];
end

function Y = qmul_right_scalar_local(X, q)
% Right multiplication: X * q.

a = X(:,:,1); b = X(:,:,2); c = X(:,:,3); d = X(:,:,4);
e = q(1); f = q(2); g = q(3);  h = q(4);

Y = zeros(size(X));
Y(:,:,1) = a*e - b*f - c*g - d*h;
Y(:,:,2) = a*f + b*e + c*h - d*g;
Y(:,:,3) = a*g - b*h + c*e + d*f;
Y(:,:,4) = a*h + b*g - c*f + d*e;
end

function Y = qmul_left_scalar_local(q, X)
%left multiplication: q * X

e = q(1); f = q(2);  g = q(3); h = q(4);
a = X(:,:,1); b = X(:,:,2); c = X(:,:,3); d = X(:,:,4);

Y = zeros(size(X));
Y(:,:,1) = e*a - f*b - g*c - h*d;
Y(:,:,2) = e*b + f*a + g*d - h*c;
Y(:,:,3) = e*c - f*d + g*a + h*b;
Y(:,:,4) = e*d + f*c - g*b + h*a;
end
