clear 
close all force
addpath('../../nn_gui');
addpath('../../nn_core');
addpath('../../nn_core/cuda');
addpath('../../nn_core/mmx');
addpath('../../nn_core/Optimizers');
addpath('../../nn_core/Activations');
addpath('../../nn_core/Activations');
addpath('../../nn_core/Wrappers');
addpath('../../nn_core/ConvNet');

PRECISION = 'double';
     % definitions(PRECISION, useGPU, whichThreads, plotOn) 
defs = definitions(PRECISION, true, [1], true); 

% Load the MNIST training set
load '~/mnist_full.mat'

%% Parameters that determine the features / input layer %%
input_size = size(X,1);
output_size = input_size;

%% Parameters defining the neural network %%
% Note: The optional third parameter to the activation function is the
%       learning rate used for pre-training of that layer.
layers.af{1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
layers.sz{1} = [input_size 1 1];
layers.typ{1} = defs.TYPES.INPUT;

layers.af{end+1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
layers.sz{end+1} = [1000 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
layers.sz{end+1} = [500 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
layers.sz{end+1} = [250 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = LinU(defs, defs.COSTS.SQUARED_ERROR);
layers.sz{end+1} = [30 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
layers.sz{end+1} = [250 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
layers.sz{end+1} = [500 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
layers.sz{end+1} = [1000 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
layers.sz{end+1} = [output_size 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

if defs.plotOn
    nnShow(423, layers, defs);
end

params = struct();
params.maxIter = precision(10000,defs);
params.momentum = precision(0.9,defs);
params.maxnorm = precision(0,defs);
params.lambda = precision(0,defs);
params.alphaTau = precision(0.5*params.maxIter,defs);
params.denoise = precision(0,defs);
params.dropout = precision(0.6,defs);
params.miniBatchSize = precision(100,defs); % set to zero to disable mini-batches
params.alpha = precision(0.01,defs); % If this is non-zero, use this learning rate for the entire network
params.rho = precision(0.95, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.eps = precision(1e-6, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.tieWeights = false;

params.beta_s = 0; % Strength of sparsity penalty; set to 0 to disable
params.rho_s0 = 0; % Target average hidden unit activation for sparsity penalty
%% 

%%%%%%%%%%%%%%%%%%%%% Pre-training Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lower layers (unsupervised pre-training):
PTparams = struct();
PTparams.maxIter = precision(500,defs);
PTparams.momentum = precision(0.8,defs);
PTparams.maxnorm = precision(0,defs);
PTparams.lambda = precision(0,defs);
PTparams.alphaTau = precision(0.5*PTparams.maxIter,defs);
PTparams.denoise = precision(0.25,defs);
PTparams.dropout = precision(1,defs);
PTparams.miniBatchSize = params.miniBatchSize; % set to zero to disable mini-batches
PTparams.beta_s = 0; % Strength of sparsity penalty; set to 0 to disable
PTparams.rho_s0 = 0; % Target average hidden unit activation for sparsity penalty

% top layer (supervised pre-training):
PTTparams = struct();
PTTparams.maxIter = precision(500,defs);
PTTparams.momentum = precision(0.8,defs);
PTTparams.maxnorm = precision(0,defs);
PTTparams.lambda = precision(0,defs);
PTTparams.alphaTau = precision(0.5*PTTparams.maxIter,defs);
PTTparams.denoise = precision(0.25,defs);
PTTparams.dropout = precision(1,defs);
PTTparams.miniBatchSize = params.miniBatchSize; % set to zero to disable mini-batches
PTTparams.beta_s = 0; % Strength of sparsity penalty; set to 0 to disable
PTTparams.rho_s0 = 0; % Target average hidden unit activation for sparsity penalty
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = precision(X,defs); % to be consistent with standard notiation

m = size(X,2); % number of examples (x_i)
N_l = uint32(numel(layers.sz));

X = varObj(X,defs, defs.TYPES.INPUT);

% Generate the class that defines the neural network
nn = nnLayers(params, layers, X, X, {}, {}, defs);
nn.initWeightsBiases();

% Pre-training
nn = nnPretrain(nn, PTparams, PTTparams, defs);
Wpt = nn.W;
bpt = nn.b;

nn.W = Wpt;
nn.b = bpt;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
costFunc = @(nn,r,newRandGen) nnCostFunctionCNN(nn,r,newRandGen); 
%profile on
%nn = gradientDescent(costFunc, nn, defs, [], [], [], [], 'Training Entire Network');
nn = gradientDescentAdaDelta(costFunc, nn, defs, [], [], [], [], 'Training Entire Network');
%profile off
%profile viewer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imgsz = sqrt(size(X.v,1));
% show the input and output of autoencoder
figure(2)
nn.A{1} = X;
A = gather(feedForward(nn, m, false, true));
idx = randperm(m);
for i=1:m
	subplot(121)
    imagesc(reshape(X.v(:,idx(i)), imgsz, imgsz), [0 1]);
    title('Original Image');
    axis square
    colormap(gray);

    subplot(122)
    imagesc(reshape(A(:,idx(i)), imgsz, imgsz), [0 1]);
    title('Reconstruction');
    axis square
    colormap(gray);
    
    waitforbuttonpress;
end

% Show the features learned
figs = double(ceil(sqrt(nn.l.sz{2})));
for i=1:double(nn.l.sz{2})
    %subplot(figs, figs+1, i);
    figure(100+i)
    imagesc(reshape(nn.W{1}(i,:), imgsz, imgsz));
    set(gca,'Xtick',[],'Ytick',[]);
    axis square
    colormap(gray);
    title(['1st Level Feature ' num2str(i)]);
    
    waitforbuttonpress;
end
