function A = dropoutLayer(A, nn, k, newRandGen, test)
% dropoutLayer: perform dropout on a particular layer AND TIME SLICE.
% k: layer number to drop out
% t: time step for recurrent net (only generate new random dropouts for first time step)
% newRandGen: generate a new dropout network, or reuse prior?
% test: is this for training or are we doing inference?
    
    if((nn.p.dropout~=1) && (k<nn.N_l) && (k>1)) % Don't dropout the output layer or the input layer
        % Just testing (not training), so scale activation by p
        if test
            A = nn.p.dropout*A;
        else
            % Generate a new set of dropout masks
            if newRandGen
                if nn.l.typ{k} == nn.defs.TYPES.CONVOLUTIONAL
                    % dropout entire image maps
                    nn.Vdo{k} = randWrapper([1, 1, size(A, 3) size(A, 4)], nn.defs) < nn.p.dropout;
                else
                    nn.Vdo{k} = randWrapper([size(A, 1) size(A, 2)], nn.defs) < nn.p.dropout;
                end
            end
            
            % Apply the dropout mask to the activations
            if  (nn.l.typ{k} == nn.defs.TYPES.LSTM) && ~isequal(size(A), size(nn.Vdo{k}))
                % Because of the asymmetry of LSTM (one output and four
                % inputs per unit), all of the deltas (error terms) coming
                % back from a single output unit must be zeroed out
                A = bsxfun(@times, A, repmat(nn.Vdo{k}, 4, 1));
            elseif nn.l.typ{k} ~= nn.defs.TYPES.AVERAGE_POOLING
            	A = bsxfun(@times, A, nn.Vdo{k});
            end
        end
    end
end

