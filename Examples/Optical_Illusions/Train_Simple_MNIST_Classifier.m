clear 
close all force
addpath('../../nn_gui');
addpath('../../nn_core');
addpath('../../nn_core/cuda');
addpath('../../nn_core/mmx');
addpath('../../nn_core/Optimizers');
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
layers.af{1} = [];
layers.sz{1} = [input_size 1 1];
layers.typ{1} = defs.TYPES.INPUT;

layers.af{end+1} = tanh_af(defs, []);
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
params.maxIter = precision(2000,defs);
params.momentum = precision(0.9,defs);
params.maxnorm = precision(0,defs);
params.lambda = precision(0,defs);
params.alphaTau = precision(0.25*params.maxIter,defs); % alpha_i = alpha*tau/(tau+i) (see "A Stochastic Quasi-Newton Method for Online Convex Optimization", Eqn. 7)
params.denoise = precision(0,defs); % set to 0 to disable
params.dropout = precision(1,defs); % set to 1 to disable
params.miniBatchSize = precision(100,defs); % set to zero to disable mini-batches
params.alpha = precision(.1,defs); % If this is non-zero, use this learning rate for the entire network
params.rho = precision(0.95, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.eps = precision(1e-6, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.tieWeights = false;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
costFunc = @(nn,r,newRandGen) nnCostFunctionCNN(nn,r,newRandGen);
nn = gradientDescentAdaDelta(costFunc, nn, defs, Xts, Yts, yts, y, 'Training Entire Network');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract the weights, biases and layers
layers = nn.l;
W = nn.W;
b = nn.b;
save('simple_mnist.mat', 'W', 'b', 'layers');

% Determine accuracy
nn.A{1} = Xts;
nn.Y = Yts;
[pred, probs]= predictDeep(feedForward(nn, size(Yts.v, 2), false, true));
acc = mean(double(pred == yts)) * 100;
fprintf('Test Set Accuracy: %g\n', acc);