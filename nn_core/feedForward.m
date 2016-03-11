function [A_N_l, J_s] = feedForward(nn, m, newRandGen, test) 
% test: are we training or testing? If testing, dropout only scales activations
% Output dimensionality of a convolutional or pooling layer is 4D: [map_x, map_y, #maps, #training examples(m)]

    J_s = 0;

    for k=2:nn.N_l
        switch nn.l.typ{k}
            case nn.defs.TYPES.FULLY_CONNECTED
                nn.A{k-1}.v = cnnFlattenLayer(nn.A{k-1}.v, m);
                nn.A{k}.v = nn.l.af{k}.activ(bsxfun(@plus, nn.b{k-1}, nn.W{k-1}*nn.A{k-1}.v));
            case nn.defs.TYPES.CONVOLUTIONAL
                nn.A{k}.v = cnnConvolve(nn.A{k-1}.v, nn.W{k-1}, nn.b{k-1}, nn, k, m);
            case nn.defs.TYPES.AVERAGE_POOLING
                nn.A{k}.v = cnnPool(nn.A{k-1}.v, nn.W{k-1}, nn.l.szo{k-1}, nn.l.szo{k}, nn.l.sz{k}, nn, k, m); 
            otherwise
                error('Unknown layer type!')
        end
        
        nn.A{k}.v = dropoutLayer(nn.A{k}.v, nn, k, newRandGen, test);
        
        % Sparsity penalty
        if ~test && (nn.p.beta_s ~= 0) && (k < nn.N_l) && (nn.l.typ{k} ~= nn.defs.TYPES.AVERAGE_POOLING)
            % Accumulate cost function penalty
            J_s = J_s + nn.p.beta_s/2/m*sum((nn.p.rho_s0 - nn.A{k}.v(:)).^2);
        end
    end

    A_N_l = nn.A{end}.v;
end
