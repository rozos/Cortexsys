%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%
clear;
close all;
addpath('../nn_gui');
addpath('../nn_core');
addpath('../nn_core/cuda');
addpath('../nn_core/mmx');
addpath('../nn_core/Optimizers');
addpath('../nn_core/Activations');
addpath('../nn_core/Wrappers');
addpath('../nn_core/ConvNet');
tic

PRECISION = 'single';
     % definitions(PRECISION, useGPU, whichThreads, plotOn) 
defs = definitions(PRECISION, true, [1], false); 

seed = posixtime(datetime); % Set random number generation seed

%Order = 4; % Use 4th order numerical check
Order = 2; % 2nd order numerical check

Nbench = 3; % Run a quick benchmark with N iterations

checkLSTM = false;
checkAutoencoder = false;
checkCNN = false;
checkCNN_LSTM = true;
checkFullyConnected = false;

params = struct();
params.lambda = precision(1,defs);
params.denoise = precision(0.25,defs);
params.dropout = precision(0.5,defs);
params.miniBatchSize = precision(0,defs); % set to zero to disable mini-batches
params.tieWeights = false;
params.beta_s = 1; % Strength of sparsity penalty; set to 0 to disable
params.rho_s0 = 0.2; % Target average hidden unit activation for sparsity penalty

%% checkLSTM defining the neural network %%
layers = struct();
if checkCNN_LSTM
    params.T = 3;
    params.Tos = 1;
    
    layers.af{1} = [];
    layers.sz{1} = [1 12 12];
    layers.typ{1} = defs.TYPES.INPUT;

    layers.af{end+1} = tanh_af(defs, []);
    layers.sz{end+1} = precision([2 5 5], defs);
    layers.typ{end+1} = defs.TYPES.CONVOLUTIONAL;
   
    layers.af{end+1} = [];
    layers.sz{end+1} = precision([2 2 2], defs);
    layers.typ{end+1} = defs.TYPES.AVERAGE_POOLING;
  
    layers.af{end+1} = tanh_af(defs, []);
    layers.sz{end+1} = [5 1 1];
    layers.typ{end+1} = defs.TYPES.RECURRENT;
    
    layers.af{end+1} = softmax(defs, defs.COSTS.CROSS_ENTROPY);
    layers.sz{end+1} = [4 1 1];
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
elseif checkLSTM
    params.T = 7;
    params.Tos = 2;
    
    layers.af{1} = [];
    layers.sz{1} = [6 1];
    layers.typ{1} = defs.TYPES.INPUT;
  
    layers.af{end+1} = tanh_af(defs, []);
    layers.sz{end+1} = [5 1 1];
    layers.typ{end+1} = defs.TYPES.RECURRENT;
    
    layers.af{end+1} = softmax(defs, defs.COSTS.CROSS_ENTROPY);
    layers.sz{end+1} = [4 1 1];
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
elseif checkCNN
    params.T = 1;
    
    layers.af{1} = [];
    layers.sz{1} = [1 28 28];
    layers.typ{1} = defs.TYPES.INPUT;

    layers.af{end+1} = tanh_af(defs, []);
    layers.sz{end+1} = precision([3 5 5], defs);
    layers.typ{end+1} = defs.TYPES.CONVOLUTIONAL;
   
    layers.af{end+1} = [];
    layers.sz{end+1} = precision([3 2 2], defs);
    layers.typ{end+1} = defs.TYPES.AVERAGE_POOLING;

    layers.af{end+1} = LReLU(defs, []);
    layers.sz{end+1} = precision([4 5 5], defs);
    layers.typ{end+1} = defs.TYPES.CONVOLUTIONAL;

    layers.af{end+1} = [];
    layers.sz{end+1} = precision([4 4 4], defs);
    layers.typ{end+1} = defs.TYPES.AVERAGE_POOLING;

    layers.af{end+1} = LReLU(defs, defs.COSTS.SQUARED_ERROR);
    layers.sz{end+1} = [4 1 1];
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

    layers.af{end+1} = softmax(defs, defs.COSTS.CROSS_ENTROPY);
    layers.sz{end+1} = [3 1 1];
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
elseif checkFullyConnected
    params.T = 1;
    
    layers = struct();
    layers.af{1} = sigmoid(defs, []);
    layers.sz{1} = [10 1 1];
    layers.typ{1} = defs.TYPES.INPUT;

    layers.af{end+1} = LinU(defs, []);
    layers.sz{end+1} = [9 1 1];
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
    
    layers.af{end+1} = LReLU(defs, []);
    layers.sz{end+1} = [8 1 1];
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
    
    layers.af{end+1} = softplus(defs, []);
    layers.sz{end+1} = [7 1 1];
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
    
    layers.af{end+1} = softmax(defs, defs.COSTS.CROSS_ENTROPY);
    layers.sz{end+1} = [6 1 1];
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
elseif checkAutoencoder
    params.T = 1;
    params.tieWeights = true;
    
    layers = struct();
    layers.af{1} = sigmoid(defs, defs.COSTS.LOGISTIC_REGRESSION);
    layers.sz{1} = precision([10 1 1], defs);
    layers.typ{1} = defs.TYPES.INPUT;

    layers.af{end+1} = LinU(defs, defs.COSTS.SQUARED_ERROR);
    layers.sz{end+1} = precision([20 1 1], defs);
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
    
    layers.af{end+1} = layers.af{1};
    layers.sz{end+1} = layers.sz{1};
    layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
end

m = 5; % number of training examples (x_i)
r = 1:m;
N_l = numel(layers.af);

% generate random input data, X
if checkFullyConnected
    X  = varObj(.1*randnWrapper([layers.sz{1}(1), m], defs), defs, defs.TYPES.INPUT);
    
    % Generate the label set
    y  = 1 + mod(1:m, layers.sz{end}(1))';
    Y = eye(layers.sz{end}(1));
    Y = varObj(Y(y,:)', defs, defs.TYPES.OUTPUT);
elseif checkCNN
    X  = varObj(.1*randnWrapper([layers.sz{1}(2), layers.sz{1}(3), layers.sz{1}(1), m], defs), defs, defs.TYPES.INPUT);
    
    % Generate the label set
    y  = 1 + mod(1:m, layers.sz{end}(1))';
    Y = eye(layers.sz{end}(1));
    Y = varObj(Y(y,:)', defs, defs.TYPES.OUTPUT);
elseif checkLSTM || checkCNN_LSTM
    X = cell(params.T+2,1);
    if checkCNN_LSTM
        X{1} = zerosWrapper([layers.sz{1}(2), layers.sz{1}(3), layers.sz{1}(1), m], defs);
        X{end} = X{1};
        for t=2:params.T+1
           X{t} = randWrapper([layers.sz{1}(2), layers.sz{1}(3), layers.sz{1}(1), m], defs);
        end
    else
        X{1} = zerosWrapper([layers.sz{1}(1), m], defs);
        X{end} = X{1};
        for t=2:params.T+1
           X{t} = randWrapper([layers.sz{1}(1), m], defs);
        end
    end
    X = varObj(X, defs, defs.TYPES.INPUT);
    
    % Generate the label set
    Y = cell(params.T+2,1);
    Y{1} = zerosWrapper([layers.sz{end}(1), m], defs);
    Y{end} = zerosWrapper([layers.sz{end}(1), m], defs);
    for t=2:params.T+1
       Y{t} = randWrapper([layers.sz{end}(1), m], defs);
    end
    Y = varObj(Y, defs, defs.TYPES.OUTPUT);
    % normalize each target example to a sum of one to form a valid PDF. This
    % is required for valid softmax behavior.
    for t=2:params.T+1
        for ex=1:m
            Y.v{t}(:,ex) = Y.v{t}(:,ex)./sum(Y.v{t}(:,ex));
        end
    end
end
 
% Generate the class that defines the neural network and stores the
% internal values
nn = nnLayers(params, layers, X, Y, {}, {}, defs);
nn.initWeightsBiases();
nn.gpu(); % send weighs/biases over to GPU (if enabled)

% Short hand for cost function
if checkAutoencoder
    costFunc = @(nn,r,newRandGen) nnCostFunctionCNN(nn,r,newRandGen);
elseif checkFullyConnected
    costFunc = @(nn,r,newRandGen) nnCostFunctionCNN(nn,r,newRandGen);
elseif checkCNN
    costFunc = @(nn,r,newRandGen) nnCostFunctionCNN(nn,r,newRandGen);
elseif checkLSTM || checkCNN_LSTM
    costFunc = @(nn,r,newRandGen) nnCostFunctionLSTM_CNN(nn,r,newRandGen); 
end

% Run a quick benchmark
[J, dJdW, dJdB] = costFunc(nn, r, true);
%profile on
tic
for i=coder.unroll(1:Nbench)
    [J, dJdW, dJdB] = costFunc(nn, r, true);
end
if defs.useGPU
    wait(defs.GPUs{1});
end
t = toc;
%profile off
%profile viewer
disp([num2str(t/Nbench*1e3) ' ms per costFunc iteration.']);

% Perform backprop gradient computation
rng(seed);
[J, dJdW, dJdB] = costFunc(nn, r, true);

% Perform numerical gradient computation
if (Order == 2)
    [dJdW_num, dJdB_num] = computeNumericalGradient(costFunc, nn);
elseif (Order == 4)
    [dJdW_num, dJdB_num] = computeNumericalGradientOh4(costFunc, nn);
end
    
% Sum up total error
numgrad = [];
grad = [];
if checkAutoencoder
    N_l = 2;
elseif params.tieWeights
    N_l = (N_l-1)/2+1;
end

for k=1:N_l-1
    if isempty(dJdW_num{k})
        continue;
    end
    
    if ((nn.l.typ{k+1} == nn.defs.TYPES.LSTM) || (nn.l.typ{k+1} == nn.defs.TYPES.RECURRENT)); E = 2; else E = 1; end
    for e=1:E
        grad = [grad; dJdW{k,e}(:); dJdB{k}(:)];
        numgrad = [numgrad; dJdW_num{k,e}(:); dJdB_num{k}(:)];

        diffW(k,e) = (norm(dJdW_num{k,e}(:) - dJdW{k,e}(:))/norm(dJdW_num{k,e}(:) + dJdW{k,e}(:)));
        ratioW{k,e} = dJdW_num{k,e}./dJdW{k,e};
    end
        diffB(k) = (norm(dJdB_num{k}(:) - dJdB{k}(:))/norm(dJdB_num{k}(:) + dJdB{k}(:)));
        ratioB{k} = dJdB_num{k}./dJdB{k};
end

diff = norm(numgrad-grad)/norm(numgrad+grad)/m;
fprintf(['Relative Difference: %g\n\n'], diff);

fprintf('--Relative differences:--\n');
for k=1:N_l-1
    if ((nn.l.typ{k+1} == nn.defs.TYPES.LSTM) || (nn.l.typ{k+1} == nn.defs.TYPES.RECURRENT)); E = 2; else E = 1; end
        for e=1:E
            fprintf('W(%i,%i): %g; ', k, e, diffW(k,e)); 
        end
end

fprintf('\n');
for k=1:N_l-1
   fprintf('b%i: %g; ', k, diffB(k)); 
end
fprintf('\n');

switch PRECISION
    case 'single'
        errThresh = 5e-2;
    case 'double'
        errThresh = 1e-9;
end
if diff/params.T > errThresh
    error('Gradient check error!');
end
