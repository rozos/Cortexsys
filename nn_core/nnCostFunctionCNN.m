function [J, dJdW, dJdB] = nnCostFunctionCNN(nn, r, newRandGen)
m = numel(r);
Y = varObj(nn.Y.getmb(r), nn.defs, nn.defs.TYPES.OUTPUT);
X = varObj(nn.X.getmb(r), nn.defs, nn.defs.TYPES.INPUT);
nn.A{1} = X;

% If denoising is turned on, dropout random inputs
if (nn.p.denoise > 0)
    if newRandGen
        nn.Mdn = randWrapper(size(nn.A{1}.v), nn.defs) > nn.p.denoise;
    end
    % zero out random inputs to the NN's first layer nn.X
    nn.A{1}.v = nn.Mdn.*nn.A{1}.v;
end

% Tie weights if option enabled
% Note: weights of CNN cannot be tied
if nn.p.tieWeights
    for k=(nn.N_l-1)/2+1:nn.N_l-1
        % Mirror weights from 1st half to 2nd half
        nn.W{k} = nn.W{nn.N_l-k}';
    end
end

%% %%%%%%%%%%%%%%%%%%%%% FEEDFORWARD PASS %%%%%%%%%%%%%%%%%%%%%%%%%%
[~, J_s] = feedForward(nn, m, newRandGen, false);

%% %%%%%%%%%%%%%%%%%% COMPUTE COST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J = nn.l.af{nn.N_l}.cost(Y, nn.A{nn.N_l}, m, 1) + J_s;

% Regularization by lambda (L2 regularization)
if (nn.p.lambda ~= 0)
    reg = 0; % L2 (squared) regularization term (sum of all weights)
    for k=1:nn.N_l-1
        reg = reg + sum(nn.W{k}(:).^2);
    end
    J = J + nn.p.lambda/2/m*reg;
end

%% %%%%%%%%%%%%%%%%%%%%% BACKPROPAGATION %%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE DELTAS (ERROR TERMS) %
% Note: This could probably be cleaned up by removing the nested switch.
%       However, doing so would require some re-achitecting of the way
%       convolutional layers store their weights and identity activation functions for
%       pooling layers. Ideally, a layer doesn't need to know what the
%       layer above is. For instance, a pooling layer needs a virtual set
%       of identity matrix weights, in addition to the pooling kernel.
for k=nn.N_l:-1:2
    switch nn.l.typ{k} % switch based on type of lower layer
        case nn.defs.TYPES.FULLY_CONNECTED
            if k==nn.N_l
                % Compute output layer deltas

                if (nn.l.af{nn.N_l}.costType == nn.l.af{nn.N_l}.defs.COSTS.CUSTOM_ERROR )
                  d{k} = nn.l.af{nn.N_l}.cost( Y, nn.A{nn.N_l}, m, 1, true)...
                         .* nn.l.af{nn.N_l}.ograd(nn.A{nn.N_l}.v);
                else
                    d{k} = (nn.A{nn.N_l}.v - Y.v).* ...
                            nn.l.af{nn.N_l}.ograd(nn.A{nn.N_l}.v);
                end
            else
                % Compute hidden layer deltas
                d{k} = nn.W{k}'*d{k+1}.*nn.l.af{k}.grad(nn.A{k}.v);
            end           
        case nn.defs.TYPES.CONVOLUTIONAL
            if k==nn.N_l        % This is output layer
                if (nn.l.af{nn.N_l}.costType == nn.l.af{nn.N_l}.defs.COSTS.CUSTOM_ERROR )
                  d{k} = nn.l.af{nn.N_l}.cost( Y, nn.A{nn.N_l}, m, 1, true)...
                         .* nn.l.af{nn.N_l}.ograd(nn.A{nn.N_l}.v);
                else
                  d{k} = (nn.A{nn.N_l}.v - Y.v).* ...
                            nn.l.af{nn.N_l}.ograd(nn.A{nn.N_l}.v);
                end
                %d{k} = cnnUnflattenLayer(d{k}, nn.l.szo{k});
                d{k} = reshape(d{k}', [ nn.l.szo{k}(2), ...
                                        length(d{k})/nn.l.szo{k}(2), ...
                                        size(d{k},1), nn.l.szo{k}(3) ] );
                d{k} = permute(d{k}, [1, 4, 3, 2] );
                                        
            else
              switch nn.l.typ{k+1} % switch based on type of layer above
                  case nn.defs.TYPES.FULLY_CONNECTED
                      d{k} = nn.W{k}'*d{k+1}.*nn.l.af{k}.grad(nn.A{k}.v);
                      d{k} = cnnUnflattenLayer(d{k}, nn.l.szo{k});
                  case nn.defs.TYPES.CONVOLUTIONAL;
                      d{k+1} = cnnUnflattenLayer(d{k+1}, nn.l.szo{k+1});
                      d{k} = cnnUnconvolve(d{k+1}, nn.W{k}, nn, k, m).*nn.l.af{k}.grad(nn.A{k}.v);
                  case nn.defs.TYPES.AVERAGE_POOLING
                      d{k+1} = cnnUnflattenLayer(d{k+1}, nn.l.szo{k+1});
                      d{k} = ultrakron(d{k+1}, nn.W{k}(:,:,1)).*nn.l.af{k}.grad(nn.A{k}.v); % "blow up" the deltas into an unpooled version
              end
            end       
        case nn.defs.TYPES.AVERAGE_POOLING
            switch nn.l.typ{k+1} % switch based on type of layer above
                case nn.defs.TYPES.FULLY_CONNECTED
                    % Note: Pooling layer has no activation function
                    d{k} = nn.W{k}'*d{k+1};
                case nn.defs.TYPES.CONVOLUTIONAL
                    d{k+1} = cnnUnflattenLayer(d{k+1}, nn.l.szo{k+1});
                    d{k} = cnnUnconvolve(d{k+1}, nn.W{k},nn, k, m);
            end
        otherwise
            error('Unknown layer type!')
    end
    
    % Apply sparsity penalty term
    if (nn.p.beta_s ~= 0) && (k < nn.N_l) && (nn.l.typ{k} ~= nn.defs.TYPES.AVERAGE_POOLING)
        d{k} = d{k} - nn.p.beta_s*(nn.p.rho_s0 - nn.A{k}.v).*nn.l.af{k}.grad(nn.A{k}.v);
    end
    
    d{k} = dropoutLayer(d{k}, nn, k, false, false);
end

% COMPUTE GRADIENTS %
for k=nn.N_l-1:-1:1
    switch nn.l.typ{k+1}
        case nn.defs.TYPES.FULLY_CONNECTED
            dJdW{k,1} = 1/m*(d{k+1}*nn.A{k}.v' + nn.p.lambda*nn.W{k,1}); 
            dJdB{k} = 1/m*sum(d{k+1}, 2);
        case nn.defs.TYPES.CONVOLUTIONAL
            [dJdW{k,1}, dJdB{k}] = cnnConvGrad(d{k+1}, nn.A{k}.v, nn.W{k}, nn, k, m);
            dJdW{k,1} = dJdW{k} + 1/m*nn.p.lambda*nn.W{k,1};
        case nn.defs.TYPES.AVERAGE_POOLING
            % No update required
        otherwise
            error('Unknown layer type!')
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gradients of tied weights are sum of W and W'
if nn.p.tieWeights
    for k=(nn.N_l-1)/2+1:nn.N_l-1
        % Mirror weights from 1st half to 2nd half
        dJdW{nn.N_l-k, 1} = dJdW{nn.N_l-k} + dJdW{k}';
        dJdW{k, 1} = dJdW{nn.N_l-k}';
    end
end

end
