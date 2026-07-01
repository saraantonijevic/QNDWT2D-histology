function Results = RunSVMRadialQNDWT2(summaryFile, outDir, nRepeats, outerK, innerK, nPermutations)
% Status coding:
%0 = Normal
%1 = Benign
%2 = InSitu
%3 = Invasive
%
if nargin < 1 || isempty(summaryFile)
    summaryFile = fullfile(pwd, 'summary2.csv');
    if ~exist(summaryFile, 'file')
        summaryFile = 'C:\Brani2\WorkTAMU\Quaternion\QNDWT2\summary2.csv';
    end
end

if nargin < 2 || isempty(outDir)
    baseDir = fileparts(summaryFile);
    if isempty(baseDir)
        baseDir = pwd;
    end
    outDir = fullfile(baseDir, 'ResultsSVM');
end

if nargin < 3 || isempty(nRepeats)
    nRepeats = 10;
end

if nargin < 4 || isempty(outerK)
    outerK = 5;
end

if nargin < 5 || isempty(innerK)
    innerK = 3;
end

if nargin < 6 || isempty(nPermutations)
    nPermutations = 1;
end

if ~exist(summaryFile, 'file')
    error('Cannot find summary file: %s', summaryFile);
end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fprintf('\nRadial SVM classification for QNDWT2 features, B = W1 A W2^* convention\n');
fprintf('Program version : QNDWT2_SUMMARY2_PERM_IMPORTANCE_GROUPS_RESULTSSVM_2026_05_13\n');
fprintf('Summary file : %s\n', summaryFile);
fprintf('Output folder: %s\n', outDir);
fprintf('Outer CV : %d-fold CV for grid tuning\n', innerK);
fprintf('Permutation repeats: %d per feature per held-out fold\n\n', nPermutations);

try
    T = readtable(summaryFile, 'TextType', 'string');
catch
    T = readtable(summaryFile);
end

if ~ismember('status', T.Properties.VariableNames)
    error('The table must contain a variable named status.');
end

y = double(T.status(:));

classCodes = [0; 1; 2; 3];
classNames = {'Normal','Benign','InSitu','Invasive'};

okRows = ismember(y, classCodes);
if any(~okRows)
    warning('Dropping %d rows with status not in {0,1,2,3}.', sum(~okRows));
    T = T(okRows,:);
    y = y(okRows);
end

[featureNames, X] = get_feature_matrix_local(T);
nFeatures = numel(featureNames);
featureGroups = define_feature_groups_local(featureNames);
nGroups = numel(featureGroups);

fprintf('Rows used : %d\n', size(X,1));
fprintf('Candidate features : %d\n', size(X,2));
fprintf('Feature groups: %d\n', nGroups);
fprintf('Convention note: H,V,D are taken from summary2.csv generated under B = W1 A W2^*\n\n');

%hyperparameter grid. KernelScale is in standardized feature units
p0 = max(size(X,2), 1);
%CGrid = [0.1, 1, 10, 100];
%scaleGrid = sqrt(p0) * [0.25, 0.5, 1, 2, 4];
CGrid = [5, 7, 10, 15, 30, 50, 100];
scaleGrid = sqrt(p0) * [0.5, 0.75, 1, 1.25, 1.6, 2, 2.5, 4];

allTrue = [];
allPred = [];
allRepeat = [];
allFold = [];
allImageIndex = [];
allFileName = {};

paramRows = struct([]);
pcount = 0;

% Permutation importance accumulators (importance = performance before
% permutation - performance after permutation)
impBalSum = zeros(nFeatures,1);
impBalSq  = zeros(nFeatures,1);
impAccSum = zeros(nFeatures,1);
impAccSq  = zeros(nFeatures,1);
impF1Sum  = zeros(nFeatures,1);
impF1Sq   = zeros(nFeatures,1);
impCount  = zeros(nFeatures,1);

grpBalSum = zeros(nGroups,1);
grpBalSq  = zeros(nGroups,1);
grpAccSum = zeros(nGroups,1);
grpAccSq = zeros(nGroups,1);
grpF1Sum = zeros(nGroups,1);
grpF1Sq = zeros(nGroups,1);
grpCount = zeros(nGroups,1);

%repeated nested cross-validation
for rr = 1:nRepeats

    set_random_seed_local(1000 + rr);
    outerCV = cvpartition(y, 'KFold', outerK);

    fprintf('Repeat %d of %d\n', rr, nRepeats);

    for ff = 1:outerK

        idxTrain = training(outerCV, ff);
        idxTest = test(outerCV, ff);

        XtrainRaw = X(idxTrain,:);
        ytrain = y(idxTrain);
        XtestRaw = X(idxTest,:);
        ytest = y(idxTest);

        [bestC, bestScale, bestInnerAcc] = tune_svm_grid_local( ...
            XtrainRaw, ytrain, CGrid, scaleGrid, innerK, classCodes);

        [Xtrain, Xtest, keptMask] = preprocess_train_test_local(XtrainRaw, XtestRaw);

        model = fit_svm_ecoc_local(Xtrain, ytrain, bestC, bestScale, classCodes);
        yhat = predict(model, Xtest);

        baseAcc = mean(yhat == ytest);
        baseBal = balanced_accuracy_local(ytest, yhat, classCodes);
        baseF1  = macro_f1_local(ytest, yhat, classCodes);

        %held-out permutation importance
        if nPermutations > 0
            keptIndex = find(keptMask);

            for jj = 1:numel(keptIndex)
                originalFeatureIndex = keptIndex(jj);

                for pp = 1:nPermutations
                    Xperm = Xtest;
                    permIndex = randperm(size(Xperm,1));
                    Xperm(:,jj) = Xperm(permIndex,jj);

                    yperm = predict(model, Xperm);

                    permAcc = mean(yperm == ytest);
                    permBal = balanced_accuracy_local(ytest, yperm, classCodes);
                    permF1  = macro_f1_local(ytest, yperm, classCodes);

                    dBal = baseBal - permBal;
                    dAcc = baseAcc - permAcc;
                    dF1 = baseF1 - permF1;

                    impBalSum(originalFeatureIndex) = impBalSum(originalFeatureIndex) + dBal;
                    impBalSq(originalFeatureIndex)  = impBalSq(originalFeatureIndex)  + dBal.^2;
                    impAccSum(originalFeatureIndex) = impAccSum(originalFeatureIndex) + dAcc;
                    impAccSq(originalFeatureIndex) = impAccSq(originalFeatureIndex)  + dAcc.^2;
                    impF1Sum(originalFeatureIndex) = impF1Sum(originalFeatureIndex)  + dF1;
                    impF1Sq(originalFeatureIndex) = impF1Sq(originalFeatureIndex)   + dF1.^2;
                    impCount(originalFeatureIndex)= impCount(originalFeatureIndex)  + 1;
                end
            end
        end

        % Held-out GROUP permutation importance; A group is permuted as a block: one permutation of test images is
        % applied simultaneously to all features in that group. This keeps the internal structure of the group intact, while disconnecting
        % the whole group from the correct image labels
        if nPermutations > 0 && nGroups > 0
            keptIndex = find(keptMask);

            for gg = 1:nGroups
                originalFeatureIndex = featureGroups(gg).featureIndex(:);
                [tf, localCols] = ismember(originalFeatureIndex, keptIndex);
                localCols = localCols(tf);
                localCols = localCols(localCols > 0);

                if isempty(localCols)
                    continue
                end

                for pp = 1:nPermutations
                    Xperm = Xtest;
                    permIndex = randperm(size(Xperm,1));
                    Xperm(:, localCols) = Xperm(permIndex, localCols);

                    yperm = predict(model, Xperm);

                    permAcc = mean(yperm == ytest);
                    permBal = balanced_accuracy_local(ytest, yperm, classCodes);
                    permF1  = macro_f1_local(ytest, yperm, classCodes);

                    dBal = baseBal - permBal;
                    dAcc = baseAcc - permAcc;
                    dF1 = baseF1  - permF1;

                    grpBalSum(gg) = grpBalSum(gg) + dBal;
                    grpBalSq(gg)  = grpBalSq(gg)  + dBal.^2;
                    grpAccSum(gg) = grpAccSum(gg) + dAcc;
                    grpAccSq(gg) = grpAccSq(gg)  + dAcc.^2;
                    grpF1Sum(gg)= grpF1Sum(gg)  + dF1;
                    grpF1Sq(gg) = grpF1Sq(gg)+ dF1.^2;
                    grpCount(gg) = grpCount(gg)  + 1;
                end
            end
        end

        allTrue = [allTrue; ytest]; %#ok<AGROW>
        allPred = [allPred; yhat]; %#ok<AGROW>
        allRepeat = [allRepeat; rr * ones(numel(ytest),1)]; %#ok<AGROW>
        allFold = [allFold; ff * ones(numel(ytest),1)]; %#ok<AGROW>

        imageIndex = find(idxTest);
        allImageIndex = [allImageIndex; imageIndex(:)]; %#ok<AGROW>

        allFileName = [allFileName; get_filename_cells_local(T, idxTest)]; %#ok<AGROW>

        pcount = pcount + 1;
        paramRows(pcount,1).repeat = rr; %#ok<AGROW>
        paramRows(pcount,1).fold = ff;
        paramRows(pcount,1).bestC = bestC;
        paramRows(pcount,1).bestKernelScale = bestScale;
        paramRows(pcount,1).bestInnerAccuracy = bestInnerAcc;
        paramRows(pcount,1).nFeaturesKept = sum(keptMask);
        paramRows(pcount,1).outerAccuracy = baseAcc;
        paramRows(pcount,1).outerBalancedAccuracy = baseBal;
        paramRows(pcount,1).outerMacroF1 = baseF1;

        fprintf('  fold %d: C = %.4g, scale = %.4g, inner acc = %.4f, outer acc = %.4f, bal acc = %.4f, macro F1 = %.4f, features = %d\n', ...
            ff, bestC, bestScale, bestInnerAcc, baseAcc, baseBal, baseF1, sum(keptMask));
    end
end

% Confusion matrix orientation
CtruePred = confusionmat(allTrue, allPred, 'Order', classCodes);
CpredTrue = CtruePred';

trueDenom = sum(CpredTrue, 1);
CtruePct = bsxfun(@rdivide, 100 * CpredTrue, max(trueDenom, 1));

overallAccuracy = sum(diag(CpredTrue)) / sum(CpredTrue(:));

trueTotals = sum(CpredTrue, 1)';
predTotals = sum(CpredTrue, 2);
diagCounts = diag(CpredTrue);

recall = diagCounts ./ max(trueTotals, 1);
precision = diagCounts ./ max(predTotals, 1);
f1 = 2 * precision .* recall ./ max(precision + recall, eps);

balancedAccuracy = mean(recall);
macroPrecision = mean(precision);
macroF1 = mean(f1);

trueVariableNames = strcat('True_', classNames);
predRowNames = strcat('Pred_', classNames);

countTable = array2table(CpredTrue, ...
    'VariableNames', trueVariableNames, ...
    'RowNames', predRowNames);

truePctTable = array2table(CtruePct, ...
    'VariableNames', trueVariableNames, ...
    'RowNames', predRowNames);

countPctCell = cell(size(CpredTrue));
for i = 1:size(CpredTrue,1)
    for j = 1:size(CpredTrue,2)
        countPctCell{i,j} = sprintf('%d (%.1f%%)', CpredTrue(i,j), CtruePct(i,j));
    end
end

countPctTable = cell2table(countPctCell, ...
    'VariableNames', trueVariableNames, ...
    'RowNames', predRowNames);

predTable = table(allRepeat, allFold, allImageIndex, allFileName, allTrue, allPred, ...
    'VariableNames', {'repeat','fold','imageIndex','fileName','trueStatus','predStatus'});

paramTable = struct2table(paramRows);

writetable(countTable, fullfile(outDir, 'svm_radial_confusion_counts.csv'), ...
    'WriteRowNames', true);
writetable(truePctTable, fullfile(outDir, 'svm_radial_confusion_truepercent.csv'), ...
    'WriteRowNames', true);
writetable(countPctTable, fullfile(outDir, 'svm_radial_confusion_count_percent.csv'), ...
    'WriteRowNames', true);
writetable(predTable, fullfile(outDir, 'svm_radial_predictions.csv'));
writetable(paramTable, fullfile(outDir, 'svm_radial_cv_parameters.csv'));

fprintf('\nAggregated 4 x 4 confusion table, counts\n');
fprintf('Rows are predicted status, columns are true status.\n');
disp(countTable)

fprintf('\nAggregated 4 x 4 confusion table, true-class percentages\n');
fprintf('Rows are predicted status, columns are true status. Columns sum to 100 percent.\n');
disp(truePctTable)

fprintf('\nAggregated 4 x 4 confusion table, count and true-class percent\n');
disp(countPctTable)

fprintf('\nPerformance summaries\n');
fprintf('Overall accuracy: %.4f\n', overallAccuracy);
fprintf('Balanced accuracy: %.4f\n', balancedAccuracy);
fprintf('Macro precision: %.4f\n', macroPrecision);
fprintf('Macro F1 : %.4f\n\n', macroF1);

perfTable = table(classCodes, classNames(:), recall, precision, f1, ...
    'VariableNames', {'status','className','recall','precision','F1'});
disp(perfTable)
writetable(perfTable, fullfile(outDir, 'svm_radial_class_performance.csv'));

%permutation importance table
importanceTable = make_importance_table_local(featureNames, impBalSum, impBalSq, ...
    impAccSum, impAccSq, impF1Sum, impF1Sq, impCount);

writetable(importanceTable, fullfile(outDir, 'svm_radial_permutation_importance.csv'));

goodPredictors = importanceTable(importanceTable.meanDropBalancedAccuracy > 0, :);
writetable(goodPredictors, fullfile(outDir, 'svm_radial_good_predictors.csv'));

topN = min(30, height(importanceTable));
goodTop30 = importanceTable(1:topN,:);
writetable(goodTop30, fullfile(outDir, 'svm_radial_good_predictors_top30.csv'));

fprintf('\nTop permutation-important predictors, ranked by drop in balanced accuracy\n');
disp(goodTop30)


% Group permutation importance table
groupImportanceTable = make_group_importance_table_local(featureGroups, grpBalSum, grpBalSq, ...
    grpAccSum, grpAccSq, grpF1Sum, grpF1Sq, grpCount);

writetable(groupImportanceTable, fullfile(outDir, 'svm_radial_group_permutation_importance.csv'));

goodFeatureGroups = groupImportanceTable(groupImportanceTable.meanDropBalancedAccuracy > 0, :);
writetable(goodFeatureGroups, fullfile(outDir, 'svm_radial_good_feature_groups.csv'));

topGroupN = min(30, size(groupImportanceTable,1));
goodFeatureGroupsTop30 = groupImportanceTable(1:topGroupN,:);
writetable(goodFeatureGroupsTop30, fullfile(outDir, 'svm_radial_good_feature_groups_top30.csv'));

fprintf('\nTop group permutation-important feature classes, ranked by drop in balanced accuracy\n');
disp(goodFeatureGroupsTop30)



%figures
try
    fig = figure('Color', 'w', 'Name', 'Radial SVM confusion table');
    imagesc(CtruePct);
    axis image
    colormap(gca, parula)
    colorbar
    set(gca, 'XTick', 1:numel(classNames), 'XTickLabel', classNames);
    set(gca, 'YTick', 1:numel(classNames), 'YTickLabel', classNames);
    xlabel('True status', 'Interpreter', 'tex')
    ylabel('Predicted status', 'Interpreter', 'tex')
    title(sprintf('Radial SVM confusion table, repeated %d x %d-fold CV', nRepeats, outerK), 'Interpreter', 'tex');

    for i = 1:size(CpredTrue,1)
        for j = 1:size(CpredTrue,2)
            text(j, i, sprintf('%d\n%.1f%%', CpredTrue(i,j), CtruePct(i,j)), ...
                'HorizontalAlignment', 'center', ...
                'FontWeight', 'bold');
        end
    end

    saveas(fig, fullfile(outDir, 'svm_radial_confusion_chart.png'));
catch ME
    warning('Could not create confusion chart: %s', ME.message);
end

try
    fig2 = figure('Color', 'w', 'Name', 'Top permutation-important predictors');
    barh(flipud(importanceTable.meanDropBalancedAccuracy(1:topN)));
    featLabels = escape_underscore_label_cellstr_local(flipud(importanceTable.feature(1:topN)));
    set(gca, 'YTick', 1:topN, ...
        'YTickLabel', featLabels, ...
        'TickLabelInterpreter', 'tex');
    xlabel('Mean decrease in held-out balanced accuracy', 'Interpreter', 'tex')
    title('Top radial-SVM permutation importance predictors', 'Interpreter', 'tex')
    grid on
    saveas(fig2, fullfile(outDir, 'svm_radial_permutation_importance_top30.png'));
catch ME
    warning('Could not create permutation importance plot: %s', ME.message);
end

try
    fig3 = figure('Color', 'w', 'Name', 'Top group permutation-important feature classes');
    barh(flipud(groupImportanceTable.meanDropBalancedAccuracy(1:topGroupN)));
    grpLabels = escape_underscore_label_cellstr_local(flipud(groupImportanceTable.groupName(1:topGroupN)));
    set(gca, 'YTick', 1:topGroupN, ...
        'YTickLabel', grpLabels, ...
        'TickLabelInterpreter', 'tex');
    xlabel('Mean decrease in held-out balanced accuracy', 'Interpreter', 'tex')
    title('Top radial-SVM group permutation importance', 'Interpreter', 'tex')
    grid on
    saveas(fig3, fullfile(outDir, 'svm_radial_group_permutation_importance_top30.png'));
catch ME
    warning('Could not create group permutation importance plot: %s', ME.message);
end

%return structure
Results = struct();
Results.summaryFile = summaryFile;
Results.outDir = outDir;
Results.nRepeats = nRepeats;
Results.outerK = outerK;
Results.innerK = innerK;
Results.nPermutations = nPermutations;
Results.featureNamesInitial = featureNames;

Results.confusionCountsPredRowsTrueCols = CpredTrue;
Results.confusionTruePercentPredRowsTrueCols = CtruePct;
Results.countTable = countTable;
Results.truePercentTable = truePctTable;
Results.countPercentTable = countPctTable;

Results.predictions = predTable;
Results.cvParameters = paramTable;
Results.classPerformance = perfTable;
Results.permutationImportance = importanceTable;
Results.goodPredictors = goodPredictors;
Results.groupPermutationImportance = groupImportanceTable;
Results.goodFeatureGroups = goodFeatureGroups;

Results.overallAccuracy = overallAccuracy;
Results.balancedAccuracy = balancedAccuracy;
Results.macroPrecision = macroPrecision;
Results.macroF1 = macroF1;

fprintf('Saved files in:\n  %s\n\n', outDir);

end



function out = escape_underscore_label_cellstr_local(labels)
% Escapes underscores for MATLAB TeX tick labels; Eg:
%   Phase_mean becomes Phase\_mean, so MATLAB prints the underscore literally instead of treating "mean" as a subscript.
%

if ischar(labels)
    labels = cellstr(labels);
elseif isstring(labels)
    labels = cellstr(labels);
end

out = labels;

for i = 1:numel(labels)
    s = labels{i};
    s = strrep(s, '_', '\_');
    out{i} = s;
end

end

%seed helper

function set_random_seed_local(seed)
% Avoids direct use of rng. This is safer on older MATLAB installations and avoids stopping if a local path shadows rng

try
    rand('twister', seed); %#ok<RAND>
catch
    try
        rand('state', seed); %#ok<RAND>
    catch
    end
end

try
    randn('state', seed); %#ok<RAND>
catch
end

end

%feature matrix

function [featureNames, X] = get_feature_matrix_local(T)

exclude = {'status', 'statusLabel', 'folder', 'fileName', 'filePath', ...
           'height', 'width', 'channelsOriginal', 'channelsUsed', ...
           'minIntensityOriginal', 'maxIntensityOriginal', ...
           'levelsRequested', 'levelsUsed', 'shift', 'error'};

names = T.Properties.VariableNames;
isNum = false(1, numel(names));

for j = 1:numel(names)
    x = T.(names{j});
    isNum(j) = isnumeric(x) || islogical(x);
end

featureNames = names(isNum);
featureNames = featureNames(~ismember(featureNames, exclude));

if isempty(featureNames)
    error('No numeric predictor features were found after excluding metadata.');
end

X = table2array(T(:, featureNames));
X = double(X);

end

function names = get_filename_cells_local(T, idx)

n = sum(idx);
if ~ismember('fileName', T.Properties.VariableNames)
    names = repmat({''}, n, 1);
    return
end

x = T.fileName(idx);

if iscell(x)
    names = x;
elseif isstring(x)
    names = cellstr(x);
elseif ischar(x)
    names = cellstr(x);
else
    names = cellstr(string(x));
end

names = names(:);

end


%feature groups for block permutation importance

function groups = define_feature_groups_local(featureNames)

groups = struct('groupName', {}, 'description', {}, 'featureIndex', {}, 'nFeatures', {});

groups = add_group_local(groups, featureNames, 'RGB_marginal', ...
    'Mean, standard deviation, and correlation of RGB channels.', ...
    {'^(mean|std)_', '^corr_'});

groups = add_group_local(groups, featureNames, 'Energy_logEnergy', ...
    'Quaternion detail energy and log-energy features, including global energy summaries.', ...
    {'^E_', '^log2E_', '^sumE_', '^share_', '^E_smooth', '^log2E_smooth', '^meanE_smooth'});

groups = add_group_local(groups, featureNames, 'Amplitude_mean', ...
    'Mean quaternion coefficient amplitudes.', ...
    {'^ampMean_'});

groups = add_group_local(groups, featureNames, 'Amplitude_std', ...
    'Standard deviation of quaternion coefficient amplitudes.', ...
    {'^ampStd_'});

groups = add_group_local(groups, featureNames, 'Amplitude_skewness', ...
    'Skewness of quaternion coefficient amplitudes.', ...
    {'^ampSkew_'});

groups = add_group_local(groups, featureNames, 'Amplitude_kurtosis', ...
    'Kurtosis of quaternion coefficient amplitudes.', ...
    {'^ampKurt_'});

groups = add_group_local(groups, featureNames, 'Amplitude_entropy', ...
    'Histogram entropy of quaternion coefficient amplitudes.', ...
    {'^ampEntropy_'});

groups = add_group_local(groups, featureNames, 'Phase_mean', ...
    'Amplitude-weighted circular mean of quaternion phase.', ...
    {'^phaseMean_'});

groups = add_group_local(groups, featureNames, 'Phase_concentration_variance', ...
    'Quaternion phase concentration and circular variance.', ...
    {'^phaseConc_', '^phaseVar_'});

groups = add_group_local(groups, featureNames, 'Axis_direction', ...
    'Quaternion phase-axis mean components and axis concentration.', ...
    {'^axisConc_', '^axisMean_'});

groups = add_group_local(groups, featureNames, 'Orientation_anisotropy', ...
    'Directional shares, H-V anisotropy, diagonal share, and orientation entropy.', ...
    {'^p_H_', '^p_V_', '^p_D_', '^A_HV_', '^A_D_', '^orientEntropy_', ...
     '^meanA_', '^stdA_', '^meanOrientEntropy', '^stdOrientEntropy'});

groups = add_group_local(groups, featureNames, 'Spectral_scaling', ...
    'Spectral entropy and slope features across scales.', ...
    {'^specEntropy_', '^slope_'});

for lev = 1:6
    groups = add_group_local(groups, featureNames, sprintf('Level_%d_all', lev), ...
        sprintf('All QNDWT features at decomposition level %d.', lev), ...
        {sprintf('_L%d$', lev)});
end

groups = add_group_local(groups, featureNames, 'Orientation_H_all', ...
    'All horizontal-direction H features and global H summaries.', ...
    {'(^|_)H(_|$)'});

groups = add_group_local(groups, featureNames, 'Orientation_V_all', ...
    'All vertical-direction V features and global V summaries.', ...
    {'(^|_)V(_|$)'});

groups = add_group_local(groups, featureNames, 'Orientation_D_all', ...
    'All diagonal-direction D features and global D summaries.', ...
    {'(^|_)D(_|$)'});

%remove accidental empty groups
keep = false(numel(groups),1);
for i = 1:numel(groups)
    keep(i) = ~isempty(groups(i).featureIndex);
end
groups = groups(keep);

end

function groups = add_group_local(groups, featureNames, groupName, description, patterns)

idx = match_feature_patterns_local(featureNames, patterns);

if isempty(idx)
    return
end

g.groupName = groupName;
g.description = description;
g.featureIndex = idx(:);
g.nFeatures = numel(idx);

groups(end+1,1) = g; %#ok<AGROW>

end

function idx = match_feature_patterns_local(featureNames, patterns)

mask = false(numel(featureNames),1);

for i = 1:numel(featureNames)
    s = featureNames{i};

    for p = 1:numel(patterns)
        if ~isempty(regexp(s, patterns{p}, 'once'))
            mask(i) = true;
            break
        end
    end
end

idx = find(mask);

end

%hyperparameter tuning

function [bestC, bestScale, bestAcc] = tune_svm_grid_local(Xraw, y, CGrid, scaleGrid, innerK, classCodes)

innerCV = cvpartition(y, 'KFold', innerK);

bestAcc = -Inf;
bestC = CGrid(1);
bestScale = scaleGrid(1);

for cc = 1:numel(CGrid)
    for ss = 1:numel(scaleGrid)

        Cnow = CGrid(cc);
        scaleNow = scaleGrid(ss);

        nCorrect = 0;
        nTotal = 0;

        for ff = 1:innerK

            idxTrain = training(innerCV, ff);
            idxVal = test(innerCV, ff);

            [XinTrain, XinVal] = preprocess_train_test_local(Xraw(idxTrain,:), Xraw(idxVal,:));

            model = fit_svm_ecoc_local(XinTrain, y(idxTrain), Cnow, scaleNow, classCodes);
            yhat = predict(model, XinVal);

            nCorrect = nCorrect + sum(yhat == y(idxVal));
            nTotal = nTotal + numel(yhat);
        end

        acc = nCorrect / nTotal;

        if acc > bestAcc
            bestAcc = acc;
            bestC = Cnow;
            bestScale = scaleNow;
        end
    end
end

end

function model = fit_svm_ecoc_local(X, y, Cnow, scaleNow, classCodes)

templ = templateSVM( ...
    'KernelFunction', 'rbf', ...
    'KernelScale', scaleNow, ...
    'BoxConstraint', Cnow, ...
    'Standardize', true);

model = fitcecoc(X, y, ...
    'Learners', templ, ...
    'Coding', 'onevsone', ...
    'ClassNames', classCodes);

end



% train-test preprocessing

function [Xtrain, Xtest, keptMask] = preprocess_train_test_local(XtrainRaw, XtestRaw)

XtrainRaw = double(XtrainRaw);
XtestRaw = double(XtestRaw);

XtrainRaw(~isfinite(XtrainRaw)) = NaN;
XtestRaw(~isfinite(XtestRaw)) = NaN;

p = size(XtrainRaw, 2);
med = zeros(1, p);

for j = 1:p
    medj = median_omitnan_local(XtrainRaw(:,j));
    if isnan(medj) || ~isfinite(medj)
        medj = 0;
    end
    med(j) = medj;
end

Xtrain = fill_nan_with_median_local(XtrainRaw, med);
Xtest = fill_nan_with_median_local(XtestRaw, med);

sd = std(Xtrain, 0, 1);
keptMask = isfinite(sd) & (sd > 1e-12);

if ~any(keptMask)
    error('All features were removed as zero variance or invalid.');
end

Xtrain = Xtrain(:, keptMask);
Xtest = Xtest(:, keptMask);

end

function m = median_omitnan_local(x)

x = x(:);
x = x(isfinite(x));
if isempty(x)
    m = NaN;
else
    m = median(x);
end

end

function X = fill_nan_with_median_local(X, med)

for j = 1:size(X,2)
    bad = isnan(X(:,j)) | ~isfinite(X(:,j));
    if any(bad)
        X(bad,j) = med(j);
    end
end

end

%performance metrics
function bal = balanced_accuracy_local(ytrue, ypred, classCodes)

C = confusionmat(ytrue, ypred, 'Order', classCodes);
den = sum(C, 2);
rec = diag(C) ./ max(den, 1);
bal = mean(rec);

end

function mf1 = macro_f1_local(ytrue, ypred, classCodes)

C = confusionmat(ytrue, ypred, 'Order', classCodes);
rec = diag(C) ./ max(sum(C,2), 1);
pre = diag(C) ./ max(sum(C,1)', 1);
f1 = 2 * pre .* rec ./ max(pre + rec, eps);
mf1 = mean(f1);

end

% permutation importance table
function groupImportanceTable = make_group_importance_table_local(featureGroups, grpBalSum, grpBalSq, ...
    grpAccSum, grpAccSq, grpF1Sum, grpF1Sq, grpCount)

n = numel(featureGroups);

groupName = cell(n,1);
description = cell(n,1);
nFeatures = zeros(n,1);

meanBal = nan(n,1);
sdBal   = nan(n,1);
meanAcc = nan(n,1);
sdAcc   = nan(n,1);
meanF1  = nan(n,1);
sdF1    = nan(n,1);

for j = 1:n
    groupName{j} = featureGroups(j).groupName;
    description{j} = featureGroups(j).description;
    nFeatures(j) = featureGroups(j).nFeatures;

    c = grpCount(j);

    if c > 0
        meanBal(j) = grpBalSum(j) / c;
        meanAcc(j) = grpAccSum(j) / c;
        meanF1(j)  = grpF1Sum(j)  / c;
    end

    if c > 1
        sdBal(j) = sqrt(max((grpBalSq(j) - grpBalSum(j)^2 / c) / (c - 1), 0));
        sdAcc(j) = sqrt(max((grpAccSq(j) - grpAccSum(j)^2 / c) / (c - 1), 0));
        sdF1(j) = sqrt(max((grpF1Sq(j)  - grpF1Sum(j)^2  / c) / (c - 1), 0));
    elseif c == 1
        sdBal(j) = 0;
        sdAcc(j) = 0;
        sdF1(j) = 0;
    end
end

sortScore = meanBal;
sortScore(~isfinite(sortScore)) = -Inf;
[~, ord] = sort(sortScore, 'descend');

rank = (1:n)';
groupImportanceTable = table(rank, groupName(ord), description(ord), nFeatures(ord), ...
    meanBal(ord), sdBal(ord), ...
    meanAcc(ord), sdAcc(ord), ...
    meanF1(ord), sdF1(ord), ...
    grpCount(ord), ...
    'VariableNames', {'rank','groupName','description','nFeatures', ...
    'meanDropBalancedAccuracy','sdDropBalancedAccuracy', ...
    'meanDropAccuracy','sdDropAccuracy', ...
    'meanDropMacroF1','sdDropMacroF1', ...
    'nContributions'});

end


function importanceTable = make_importance_table_local(featureNames, impBalSum, impBalSq, ...
    impAccSum, impAccSq, impF1Sum, impF1Sq, impCount)

n = numel(featureNames);

meanBal = nan(n,1);
sdBal = nan(n,1);
meanAcc = nan(n,1);
sdAcc= nan(n,1);
meanF1 = nan(n,1);
sdF1 = nan(n,1);

for j = 1:n
    c = impCount(j);

    if c > 0
        meanBal(j) = impBalSum(j) / c;
        meanAcc(j) = impAccSum(j) / c;
        meanF1(j) = impF1Sum(j)  / c;
    end

    if c > 1
        sdBal(j) = sqrt(max((impBalSq(j) - impBalSum(j)^2 / c) / (c - 1), 0));
        sdAcc(j) = sqrt(max((impAccSq(j) - impAccSum(j)^2 / c) / (c - 1), 0));
        sdF1(j) = sqrt(max((impF1Sq(j)  - impF1Sum(j)^2  / c) / (c - 1), 0));
    elseif c == 1
        sdBal(j) = 0;
        sdAcc(j) = 0;
        sdF1(j) = 0;
    end
end

sortScore = meanBal;
sortScore(~isfinite(sortScore)) = -Inf;
[~, ord] = sort(sortScore, 'descend');

rank = (1:n)';
importanceTable = table(rank, featureNames(ord)', ...
    meanBal(ord), sdBal(ord), ...
    meanAcc(ord), sdAcc(ord), ...
    meanF1(ord), sdF1(ord), ...
    impCount(ord), ...
    'VariableNames', {'rank','feature', ...
    'meanDropBalancedAccuracy','sdDropBalancedAccuracy', ...
    'meanDropAccuracy','sdDropAccuracy', ...
    'meanDropMacroF1','sdDropMacroF1', ...
    'nContributions'});

end
