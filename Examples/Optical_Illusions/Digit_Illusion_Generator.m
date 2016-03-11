% This script trains a simple classifier, and then generates "optical
% illusions" for the neural net by maximizing the activations by optimizing
% the inputs to the network.
clear 
close all force
addpath('../../nn_gui');
addpath('../../nn_core');
addpath('../../nn_core/Optimizers');
addpath('../../nn_core/Activations');
addpath('../../nn_core/Wrappers');

PRECISION = 'double';
     % definitions(PRECISION, useGPU, whichThreads, plotOn) 
defs = definitions(PRECISION, true, [1], true); 

% Load the pre-trained classifier
load simple_mnist

% Settings for maximizing the output activations
params.alpha = .1;
params.momentum = 0.95;
params.maxIter = 10000;
params.alphaTau = 0.25*params.maxIter;
params.lambda = .1;
params.dropout = 1;

% Construct a new network from the trained parameters
X = varObj([],defs, defs.TYPES.INPUT);
nn = nnLayers(params, layers, X, [], W, b, defs);

% Generate one illusion for each class at the network output
for Ni = 1:layers.sz{end}(1)
    %% Seed the illusion with some random noise on (0,1)
    X.v = rand(layers.sz{1}(1), 1);
    [Abest, Xbest] = gradientAscentActivations(nn, Ni, defs, ['Maximizing Activation for Output ' num2str(Ni)]);

    nn.X.v = Xbest;
    A = feedForward(nn, 1, false, true);

    figure(123);
    movegui('west');
    subplot(4,3,Ni)
    imagesc(reshape(Xbest, 28, 28))
    set(gca,'YTickLabel','')
    set(gca,'XTickLabel','')
    title(['No. ' num2str(Ni) ', conf: ' num2str(A(Ni)*100,4) '%']);
    colormap gray
end