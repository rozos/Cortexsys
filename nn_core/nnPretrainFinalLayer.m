% Take in a neural network object and train it layer-by-layer using the 
% tied-weight auto-encoder method
function nn = nnPretrainFinalLayer(nn, A0, params, defs)  
        %pre-train final layer in a supervised fashion
        disp(['Pre-training output layer ' num2str(nn.N_l) '.']);
        
        k = nn.N_l;
        
        params.alpha = nn.l.af{k}.alpha;
        params.eps = nn.l.af{k}.eps;
        params.rho = nn.l.af{k}.rho;
        params.tieWeights = false;
        
        % Construct a neural network representation describing the
        % auto-encoder (input, hidden and input' layer)
        layers = struct();
        layers.af{1} = nn.l.af{k-1};
        layers.sz{1} = nn.l.sz{k-1};
        layers.typ{1} = defs.TYPES.INPUT;
        
        layers.af{2} = nn.l.af{k};
        layers.sz{2} = nn.l.sz{k};
        layers.typ{2} = defs.TYPES.FULLY_CONNECTED;
            
        nn_k = nnLayers(params, layers, A0, nn.Y, {}, {}, defs);
        nn_k.initWeightsBiases();
        
        costFunc = @(nn_kp,r,newRandGen) nnCostFunctionCNN(nn_kp,r,newRandGen); 
        nn_k = gradientDescentAdaDelta(costFunc, nn_k, defs, [], [], [], [], ['Pre-training output layer (' num2str(k) ')']);
        
        % Save trained weights into the main NN
        nn.W{k-1} = nn_k.W{1};
        nn.b{k-1} = nn_k.b{1};
end