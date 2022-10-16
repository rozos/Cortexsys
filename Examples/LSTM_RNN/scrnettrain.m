clear;
close all force;
addpath('../../nn_gui');
addpath('../../nn_core');
addpath('../../nn_core/cuda');
addpath('../../nn_core/mmx');
addpath('../../nn_core/Optimizers');
addpath('../../nn_core/Activations');
addpath('../../nn_core/Activations');
addpath('../../nn_core/Wrappers');
addpath('../../octave_wrappers');
addpath('../../nn_core/ConvNet');
addpath('Text');

PRECISION = 'double';

     % definitions(PRECISION, useGPU, whichThreads, plotOn) 
defs = definitions(PRECISION, false, [1], false);

% Load the Shakespeares training set
Nchars = input('How many characters long for each sequence =')
if isempty(Nchars)
  Nchars = 50; % How many characters long for each sequence 
end
offset = 0; % How much to shift the labels from the input data
txtpath = 'shakespeare_subset.txt';
[X, vmap] = streamText2mat(txtpath, Nchars, offset);
offset = 1;
[Y, ~] = streamText2mat(txtpath, Nchars, offset);
                        
input_size = size(X{1},1);
output_size = input_size;
T = numel(X); % Length of all time sequences

% Both X and Y must include T=0 and T=Tf+1 'boundary conditions' filled
% with zeros for convenience
X(2:end+1) = X(:); X{1} = 0*X{1};
X(end+1) = X(1);

Y(2:end+1) = Y(:); Y{1} = 0*Y{1};
Y(end+1) = Y(1);

%%%%%%%%%%%%%%%%%%%%% Fine tuning Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params = struct();
params.maxIter = precision(5000,defs);
params.momentum = precision(0.9,defs);
params.maxnorm = precision(0,defs);
params.lambda = precision(0,defs);
params.alphaTau = precision(0.25*params.maxIter,defs); % alpha_i = alpha*tau/(tau+i) (see "A Stochastic Quasi-Newton Method for Online Convex Optimization", Eqn. 7)
params.denoise = precision(0,defs); % set to 0 to disable
params.dropout = precision(0.6,defs); % set to 1 to disable
params.miniBatchSize = precision(50,defs); % set to zero to disable mini-batches
params.tieWeights = false;
params.T = T;
params.Tos = 0; % This is the "offset" time before the cost starts accumulating for the LSTM output
                 % The idea here is that the LSTM can be fed with inputs
                 % for n time steps, and won't be penalized for predictions
                 % until t>=Tos. This helps by giving the LSTM context.
% Optimization routine parameters:     
params.alpha = precision(.001,defs); % If this is non-zero, use this learning rate for the entire network
params.rho = precision(0.95, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.eps = precision(1e-6, defs); % AdaDelta hyperparameter (don't generally need to modify)
params.cg.N = 10; % Max CG iterations before reset
params.cg.sigma0 = 0.01; % CG Secant step-method parameter
params.cg.jmax = 10; % Maximum CG Secant iterations
params.cg.eps = 1e-4; % Update threshold for CG
params.cg.mbIters = 10; % How many CG iterations per minibatch?
                 
%%%%%%%%%%%%%%%%%%%%%%%%% Layer Setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
layers.af{1} = [];
layers.sz{1} = [input_size 1 1];
layers.typ{1} = defs.TYPES.INPUT;

layers.af{end+1} = tanh_af(defs, []);
layers.sz{end+1} = [128 1 1];
layers.typ{end+1} = defs.TYPES.LSTM;

layers.af{end+1} = softmax(defs, defs.COSTS.CROSS_ENTROPY);
layers.sz{end+1} = [output_size 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

if defs.plotOn
    nnShow(423, layers, defs);
end

% Process Y such that first time sequence is stripped off and replaced by
% a null at the end. This will cause LSTM to predict next character in the
% sequence. The final prediction for a sequence should be null (zero).

X = varObj(X,defs,defs.TYPES.INPUT);
Y = varObj(Y,defs,defs.TYPES.OUTPUT);

nn = nnLayers(params, layers, X, Y, {}, {}, defs);
nn.initWeightsBiases();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
costFunc = @(nn,r,newRandGen) nnCostFunctionLSTM(nn,r,newRandGen); 
nn = gradientDescentAdaDelta(costFunc, nn, defs, [], [], [], [], 'Training Entire Network');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
