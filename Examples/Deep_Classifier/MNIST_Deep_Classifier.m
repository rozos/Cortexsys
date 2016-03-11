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
%%

% Load the MNIST training set
load '~/mnist_full.mat'

y(y == 0) = 10; % make zero into a ten for indexing purposes
yts(yts == 0) = 10;

%% Parameters that determine the features / input layer %%
input_size = size(X,1);
output_size = 10;

%% Parameters defining the neural network %%
% Note: The optional third parameter to the activation function is the
%       learning rate used for pre-training of that layer.
layers.af{1} = LReLU(defs, defs.COSTS.SQUARED_ERROR);
layers.sz{1} = [input_size 1 1];
layers.typ{1} = defs.TYPES.INPUT;

layers.af{end+1} = LReLU(defs, defs.COSTS.SQUARED_ERROR);
layers.sz{end+1} = [768 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = LReLU(defs, defs.COSTS.SQUARED_ERROR);
layers.sz{end+1} = [256 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = softmax(defs, defs.COSTS.CROSS_ENTROPY);
layers.sz{end+1} = [output_size 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

if defs.plotOn
    nnShow(423, layers, defs);
end

%%%%%%%%%%%%%%%%%%%%% Fine tuning Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params = struct();
params.maxIter = precision(10000,defs);
params.momentum = precision(0.9,defs);
params.maxnorm = precision(0,defs);
params.lambda = precision(0,defs);
params.alphaTau = precision(0.25*params.maxIter,defs); % alpha_i = alpha*tau/(tau+i) (see "A Stochastic Quasi-Newton Method for Online Convex Optimization", Eqn. 7)
params.denoise = precision(0.25,defs); % set to 0 to disable
params.dropout = precision(0.6,defs); % set to 1 to disable
params.miniBatchSize = precision(100,defs); % set to zero to disable mini-batches
params.alpha = precision(.1,defs); % If this is non-zero, use this learning rate for the entire network
params.rho = precision(0.95, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.eps = precision(1e-6, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.tieWeights = false;

params.beta_s = 0; % Strength of sparsity penalty; set to 0 to disable
params.rho_s0 = 0; % Target average hidden unit activation for sparsity penalty

params.cg.N = 10; % Max CG iterations before reset
params.cg.sigma0 = 0.01; % CG Secant step-method parameter
params.cg.jmax = 10; % Maximum CG Secant iterations
params.cg.eps = 1e-4; % Update threshold for CG
params.cg.mbIters = 10; % How many CG iterations per minibatch?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%% Pre-training Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lower layers (unsupervised pre-training):
PTparams = struct();
PTparams.maxIter = precision(300,defs);
PTparams.momentum = precision(0.8,defs);
PTparams.maxnorm = precision(0,defs);
PTparams.lambda = precision(0,defs);
PTparams.alphaTau = precision(0.5*params.maxIter,defs);
PTparams.denoise = precision(0.25,defs);
PTparams.dropout = precision(1,defs);
PTparams.miniBatchSize = params.miniBatchSize; % set to zero to disable mini-batches
PTparams.beta_s = 0; % Strength of sparsity penalty; set to 0 to disable
PTparams.rho_s0 = 0; % Target average hidden unit activation for sparsity penalty

% top layer (supervised pre-training):
PTTparams = struct();
PTTparams.maxIter = precision(300,defs);
PTTparams.momentum = precision(0.8,defs);
PTTparams.maxnorm = precision(0,defs);
PTTparams.lambda = precision(0,defs);
PTTparams.alphaTau = precision(0.5*params.maxIter,defs);
PTTparams.denoise = precision(0.25,defs);
PTTparams.dropout = precision(1,defs);
PTTparams.miniBatchSize = params.miniBatchSize; % set to zero to disable mini-batches
PTTparams.beta_s = 0; % Strength of sparsity penalty; set to 0 to disable
PTTparams.rho_s0 = 0; % Target average hidden unit activation for sparsity penalty
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = precision(X,defs); % to be consistent with standard notiation

%% classifier 
Y = speye(layers.sz{end}(1));
Y = Y(y,:)';

Yts = speye(layers.sz{end}(1));
Yts = Yts(yts,:)';

Xts = varObj(precision(Xts, defs), defs, defs.TYPES.INPUT);
Yts = varObj(Yts, defs, defs.TYPES.OUTPUT);

m = size(X,2); % number of examples (x_i)
N_l = numel(layers.sz);

X = varObj(X,defs, defs.TYPES.INPUT);
Y = varObj(Y,defs, defs.TYPES.OUTPUT);
nn = nnLayers(params, layers, X, Y, {}, {}, defs);
nn.initWeightsBiases();

% Pre-training
nn = nnPretrain(nn, PTparams, PTTparams, defs);
% Store pre-trained parameters
Wpt = nn.W;
bpt = nn.b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
costFunc = @(nn,r,newRandGen) nnCostFunctionCNN(nn,r,newRandGen); 
%profile on
%nn = gradientDescent(costFunc, nn, defs, Xts, Yts, yts, y, 'Training Entire Network');
nn = gradientDescentAdaDelta(costFunc, nn, defs, Xts, Yts, yts, y, 'Training Entire Network');
%profile off
%profile viewer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Determine accuracy
nn.A{1} = Xts;
nn.Y = Yts;
[pred, probs]= predictDeep(feedForward(nn, size(Yts.v, 2), false, true));
acc = mean(double(pred == yts)) * 100;
fprintf('Test Set Accuracy: %g\n', acc);

%% Post-analysis, plotting, etc.
% Determine all errors and display them
figure
errs = find(pred~=yts); % the training example index with an error
for i=1:numel(errs)
   imagesc(reshape(Xts.v(:,errs(i)), sqrt(input_size), sqrt(input_size)));
   colormap(gray);
   title(['Predicted: ' num2str(pred(errs(i))) ', Actual: ' num2str(yts(errs(i)))]);
   waitforbuttonpress;
end