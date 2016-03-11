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

%%%%%%%%%%%%%%%%%%%%% Training Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params = struct();
params.maxIter = precision(10000,defs);
params.maxnorm = precision(0,defs);
params.lambda = precision(.05,defs);
params.alphaTau = precision(0.25*params.maxIter,defs); % alpha_i = alpha*tau/(tau+i) (see "A Stochastic Quasi-Newton Method for Online Convex Optimization", Eqn. 7)
params.denoise = precision(0.25,defs); % set to 0 to disable
params.dropout = precision(0.6,defs); % set to 1 to disable
params.miniBatchSize = precision(128,defs); % set to zero to disable mini-batches
params.autoenc = false;
params.tieWeights = false;

params.beta_s = 0; % Strength of sparsity penalty; set to 0 to disable
params.rho_s0 = 0; % Target average hidden unit activation for sparsity penalty

params.momentum = precision(0.9,defs);
params.alpha = precision(.01,defs); % If this is non-zero, use this learning rate for the entire network
params.rho = precision(0.95, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.eps = precision(1e-6, defs); % AdaDelta hyperparameter (don't generally need to modify)

%%%%%%%%%%%%%%%%%%%%%%%% Network Topology %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
layers.af{1} = [];
layers.sz{1} = [1 28 28];
layers.typ{1} = defs.TYPES.INPUT;

layers.af{end+1} = LReLU(defs, []);
layers.sz{end+1} = precision([8 5 5], defs);
layers.typ{end+1} = defs.TYPES.CONVOLUTIONAL;

layers.af{end+1} = LinU(defs, []);
layers.sz{end+1} = precision([8 2 2], defs);
layers.typ{end+1} = defs.TYPES.AVERAGE_POOLING;

layers.af{end+1} = LReLU(defs, []);
layers.sz{end+1} = precision([12 5 5], defs);
layers.typ{end+1} = defs.TYPES.CONVOLUTIONAL;

layers.af{end+1} = LinU(defs, []);
layers.sz{end+1} = precision([12 2 2], defs);
layers.typ{end+1} = defs.TYPES.AVERAGE_POOLING;

layers.af{end+1} = softmax(defs, defs.COSTS.CROSS_ENTROPY);
layers.sz{end+1} = [10 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

if defs.plotOn
    nnShow(423, layers, defs);
end

%%%%%%%%%%%%%%%%%% Setup Data and Target Labels %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct target label matrix
Y = speye(layers.sz{end}(1));
Y = Y(y,:)';
Y = varObj(Y, defs, defs.TYPES.OUTPUT);

% Reshape MNIST image data into 28x28 images (grayscale / 1 input map per image)
X = varObj(reshape(X, [28 28 1 size(X,2)]), defs, defs.TYPES.INPUT);

% Construct target label matrix
Yts = speye(layers.sz{end}(1));
Yts = Yts(yts,:)';
Yts = varObj(Yts, defs, defs.TYPES.OUTPUT);

% Reshape MNIST image data into 28x28 images (grayscale / 1 input map per image)
Xts = varObj(reshape(Xts, [28 28 1 size(Xts,2)]), defs, defs.TYPES.INPUT);

%%%%%%%%%%%%%%%%%%%%%%%% Setup Neural Net Object %%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate the class that defines the neural network and stores the internal values
nn = nnLayers(params, layers, X, Y, {}, {}, defs);
nn.initWeightsBiases();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
costFunc = @(nn,r,newRandGen) nnCostFunctionCNN(nn,r,newRandGen); 
nn = gradientDescentAdaDelta(costFunc, nn, defs, [], [], [], [], 'Training Entire Network');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine accuracy
nn.disableCuda(); % get everything back from GPU
nn.A{1} = Xts;
nn.Y = Yts;
[pred, probs]= predictDeep(feedForward(nn, size(Yts.v,2), false, true));
acc = mean(double(pred == yts)) * 100;
fprintf('Test Set Accuracy: %g\n', acc);

%% Post-analysis, plotting, etc.
% Determine all errors and display them
figure
errs = find(pred~=yts); % the training example index with an error
for i=1:numel(errs)
   imagesc(squeeze(Xts.v(:,:,:,errs(i))));
   colormap(gray);
   title(['Predicted: ' num2str(pred(errs(i))) ', Actual: ' num2str(yts(errs(i)))]);
   waitforbuttonpress;
end