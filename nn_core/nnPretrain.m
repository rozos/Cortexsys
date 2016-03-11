% Take in a neural network object and train it layer-by-layer
function nn = nnPretrain(nn, PTparams, PTTparams, defs)
    A2 = nn.X;
    
    %pre-train layers 2 to Nl
    for k=2:nn.N_l-1
        disp(['Pre-training layer ' num2str(k) '.']);
        
        PTparams.alpha = nn.l.af{k}.alpha;
        PTparams.eps = nn.l.af{k}.eps;
        PTparams.rho = nn.l.af{k}.rho;
        PTparams.tieWeights = true;
        
        % Construct a neural network representation describing the
        % auto-encoder (input, hidden and input' layer)
        layers = struct();
        layers.af{1} = nn.l.af{k-1};
        layers.sz{1} = nn.l.sz{k-1};
        layers.typ{1} = defs.TYPES.INPUT;
        
        layers.af{2} = nn.l.af{k};
        layers.sz{2} = nn.l.sz{k};
        layers.typ{2} = defs.TYPES.FULLY_CONNECTED;
        
        layers.af{3} = nn.l.af{k-1};
        layers.sz{3} = nn.l.sz{k-1};
        layers.typ{3} = defs.TYPES.FULLY_CONNECTED;
        
        nn_k = nnLayers(PTparams, layers, A2, A2, {}, {}, defs);
        
        % We generate inital weights and biases
        nn_k.initWeightsBiases();
        nn_k.W{2} = nn_k.W{1}'; % Start with tied weights
        
        costFunc = @(nn_kp,r,newRandGen) nnCostFunctionCNN(nn_kp,r,newRandGen); 
        nn_k = gradientDescentAdaDelta(costFunc, nn_k, defs, [], [], [], [], ['Pre-training Layer ' num2str(k)]);
        
        % Save "output" of previous layer to use as input to next layer
        A2 = varObj(gather(nn_k.A{2}), defs, defs.TYPES.INPUT);
        
        % Save trained weights into the main NN
        nn.W{k-1} = nn_k.W{1};
        nn.b{k-1} = nn_k.b{1};
        
        clear W;
        clear bias;
        clear nn_k;
    end
    
    % Train the output layer if training parameters were provided
    if ~isempty(PTTparams)
        nn = nnPretrainFinalLayer(nn, A2, PTTparams, defs);
    end
    
end