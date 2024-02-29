function [J, dJdW, dJdB] = nnCostFunctionLSTM_CNN(nn, r, newRandGen)
% r: which examples to train on (the mini-batch)
% newRandGen: generate new random values, or use prior stored values?
m = numel(r); 

% lambda: L2 regularization term
lambda = gpuArrayWrapper(nn.p.lambda,nn.defs);
Tos = nn.p.Tos; % Time offset to wait before counting output against the network's performance / cost
T = nn.p.T; % time series length for Backpropagation Through Time (BPTT)

% Extract the inputs and output data for the minibatch, convert to full
% matrix, and optionally send to GPU
Y = varObj(nn.Y.getmb(r), nn.defs, nn.defs.TYPES.OUTPUT);
X = varObj(nn.X.getmb(r), nn.defs, nn.defs.TYPES.INPUT);
nn.A{1} = X;

%% PREALLOCATE MEMORY %%
preallocateMemory(nn, m, T);

% If denoising is turned on, dropout random inputs
if (nn.p.denoise > 0)
    if newRandGen
        nn.Mdn = randWrapper(size(nn.A{1}.v), nn.defs) > nn.p.denoise;
    end
    % zero out random inputs to the NN's first layer
    nn.A{1}.v = nn.Mdn.*nn.A{1}.v;
end

%% %%%%%%%%%%%%%%%%%%%%% FEEDFORWARD PASS %%%%%%%%%%%%%%%%%%%%%%%%%%
% t = 1 corresponds to time zero (initial conditions)
%for t=coder.unroll(2:T+1, T > 10) % unroll if enough time steps
for t=2: (T+1)
    feedforwardLSTM_CNN(nn, m, t, newRandGen, false);
end

%% %%%%%%%%%%%%%%%%%% COMPUTE COST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J = nn.l.af{nn.N_l}.cost(Y, nn.A{nn.N_l}, m, 2+Tos:T+1);

% Regularization by lambda (L2 regularization)
%%% REPLACE BY CELLFUN?
if (nn.p.lambda ~= 0)
    reg = 0; % L2 (squared) regularization term (sum of all weights)
    for k=1:nn.N_l-1
        reg = reg + sum(nn.W{k}(:).^2);
    end
    J = J + lambda/2/m*reg;
end
%%

%% %%%%%%%%%%%%%%%%%%%%% BACKPROPAGATION %%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE DELTAS (ERROR TERMS) %
% Note: delta(T+2) = 0 (no errors from time outside of our sequence length)
%
% Tos_mask (T offset mask) zeros out the errors (deltas) flowing back from
% the output layer to accomodate the situation where we wish to not
% penalize the network for a given initial time (to provide context)
Tos_mask = [zeros(1+Tos,1); ones(T-Tos+1,1)];

%for t=coder.unroll(T+1:-1:2, T > 10) % unroll if enough time steps
for t=(T+1):-1:2
    for k=nn.N_l:-1:2
        switch nn.l.typ{k}
            case nn.defs.TYPES.FULLY_CONNECTED
                if k==nn.N_l
                    % Compute output layer deltas
                    % Note: nn.Y.v(:,:,t-1) corresponds to just 't' for non input/output arrays
                    nn.d{k}(:,:,t) = Tos_mask(t)*(nn.A{nn.N_l}.v(:,:,t) - Y.v(:,:,t)).*nn.l.af{nn.N_l}.ograd(nn.A{nn.N_l}.v(:,:,t));
                else
                    % Compute hidden layer deltas
                    nn.d{k}(:,:,t) = nn.W{k}'*nn.d{k+1}(:,:,t).*nn.l.af{k}.grad(nn.A{k}.v(:,:,t));
                end
            case nn.defs.TYPES.RECURRENT
                nn.d{k}(:,:,t) = nn.l.af{k}.grad(nn.A{k}.v(:,:,t)).*(nn.W{k}'*nn.d{k+1}(:,:,t) + nn.W{k-1,nn.rnn.h}'*nn.d{k}(:,:,t+1));
            case nn.defs.TYPES.LSTM
                % Cell outputs
                nn.eps_c{k}(:,:,t) = nn.W{k}'*nn.d{k+1}(:,:,t) + ...
                                     nn.W{k-1, nn.lstm.h}'*nn.d{k}(:,:,t+1);
                
                % Output gates
                nn.d{k}(nn.lstm.idx_ohm,:,t) = nn.l.af{k}.activ(nn.s_c{k}(:,:,t)).*nn.eps_c{k}(:,:,t).*nn.sigmoid_af.grad(nn.A_lstm{k}(nn.lstm.idx_ohm,:,t));
    
                % States (grad is applied to h(nn.s_c)S)
                nn.eps_s{k}(:,:,t) = nn.A_lstm{k}(nn.lstm.idx_ohm,:,t)  .* nn.l.af{k}.grad(nn.l.af{k}.activ(nn.s_c{k}(:,:,t))).*nn.eps_c{k}(:,:,t) ...
                                   + nn.A_lstm{k}(nn.lstm.idx_phi,:,t+1).* nn.eps_s{k}(:,:,t+1);
                
                % Cells
                nn.d{k}(nn.lstm.idx_c,:,t) = nn.A_lstm{k}(nn.lstm.idx_l,:,t).*nn.eps_s{k}(:,:,t).*nn.l.af{k}.grad(nn.A_lstm{k}(nn.lstm.idx_c,:,t));
                
                % Forget gates
                nn.d{k}(nn.lstm.idx_phi,:,t) = nn.s_c{k}(:,:,t-1).*nn.eps_s{k}(:,:,t).*nn.sigmoid_af.grad(nn.A_lstm{k}(nn.lstm.idx_phi,:,t));
                
                % Input gates
                nn.d{k}(nn.lstm.idx_l,:,t) = nn.A_lstm{k}(nn.lstm.idx_c,:,t).*nn.eps_s{k}(:,:,t).*nn.sigmoid_af.grad(nn.A_lstm{k}(nn.lstm.idx_l,:,t));
            case nn.defs.TYPES.CONVOLUTIONAL
                switch nn.l.typ{k+1} % switch based on type of layer above
                    case {nn.defs.TYPES.FULLY_CONNECTED, nn.defs.TYPES.LSTM, nn.defs.TYPES.RECURRENT}
                        Ak = cnnFlattenLayer(nn.A{k}.v, m, t);
                        dk = nn.W{k}'*nn.d{k+1}(:,:,t).*nn.l.af{k}.grad(Ak);
                        nn.d{k}(:,:,:,:,t) = cnnUnflattenLayer(dk, nn.l.szo{k});
                    case nn.defs.TYPES.CONVOLUTIONAL
                        nn.d{k+1}(:,:,:,:,t) = cnnUnflattenLayer(nn.d{k+1}(:,:,:,:,t), nn.l.szo{k+1});
                        nn.d{k}(:,:,:,:,t) = cnnUnconvolve(nn.d{k+1}(:,:,:,:,t), nn.W{k}, nn, k, m).*nn.l.af{k}.grad(nn.A{k}.v(:,:,:,:,t));
                    case nn.defs.TYPES.AVERAGE_POOLING
                        nn.d{k+1}(:,:,:,:,t) = cnnUnflattenLayer(nn.d{k+1}(:,:,:,:,t), nn.l.szo{k+1});
                        nn.d{k}(:,:,:,:,t) = ultrakron(nn.d{k+1}(:,:,:,:,t), nn.W{k}(:,:,1)).*nn.l.af{k}.grad(nn.A{k}.v(:,:,:,:,t)); % "blow up" the deltas into an unpooled version
                end       
            case nn.defs.TYPES.AVERAGE_POOLING
                switch nn.l.typ{k+1} % switch based on type of layer above
                    case {nn.defs.TYPES.FULLY_CONNECTED, nn.defs.TYPES.LSTM, nn.defs.TYPES.RECURRENT}
                        % Note: Pooling layer has no activation function
                        dk = nn.W{k}'*nn.d{k+1}(:,:,t);
                        nn.d{k}(:,:,:,:,t) = cnnUnflattenLayer(dk, nn.l.szo{k});
                    case nn.defs.TYPES.CONVOLUTIONAL
                        nn.d{k+1}(:,:,:,:,t) = cnnUnflattenLayer(nn.d{k+1}(:,:,t), nn.l.szo{k+1});
                        nn.d{k}(:,:,:,:,t) = cnnUnconvolve(nn.d{k+1}(:,:,:,:,t), nn.W{k},nn, k, m);
                end
            otherwise
                error('Unknown layer type!')
        end
        
        switch ndims(nn.d{k})
            case 3
                nn.d{k}(:,:,t) = dropoutLayer(nn.d{k}(:,:,t), nn, k, false, false);
            case 5
                nn.d{k}(:,:,:,:,t) = dropoutLayer(nn.d{k}(:,:,:,:,t), nn, k, false, false);
        end
        
    end
end

% COMPUTE GRADIENTS %
% Note: pagemultWrapper multiplies each 2D slice of a 3D matrix (third
%       dimension is time) together. We then sum over the 3rd time dimension to
%       accumulate total gradient.
for k=nn.N_l-1:-1:1
    switch nn.l.typ{k+1}
        case nn.defs.TYPES.FULLY_CONNECTED
            Ak = cnnFlattenLayer(nn.A{k}.v, m);
            % Note: 'nt' argument to mmx means transpose 2nd matrix (A)
            dJdW{k,1} = 1/m*(sum(pagemultWrapper(nn.d{k+1}, Ak, nn.defs), 3) + lambda*nn.W{k});
            dJdB{k} = 1/m*sum(sum(cat(3, nn.d{k+1}), 3), 2);
        case nn.defs.TYPES.RECURRENT
            Ak = cnnFlattenLayer(nn.A{k}.v, m);
            dJdW{k,nn.rnn.i} = 1/m*(sum(pagemultWrapper(nn.d{k+1}, Ak, nn.defs), 3) + lambda*nn.W{k,nn.rnn.i});
            % d*A for hidden layer is between d(t+1) and A(t)
            % Hidden weights do not incorporate L2 regularization (drops out of gradient)
            dJdW{k,nn.rnn.h} = 1/m*sum(pagemultWrapper(nn.d{k+1}(:,:,3:T+1), nn.A{k+1}.v(:,:,2:T), nn.defs), 3);
            dJdB{k} = 1/m*sum(sum(cat(3, nn.d{k+1}), 3), 2);
        case nn.defs.TYPES.LSTM
            Ak = cnnFlattenLayer(nn.A{k}.v, m);
            % Compute gradients of input weights
            dJdW{k, nn.lstm.i} = 1/m*(sum(pagemultWrapper(nn.d{k+1}, Ak, nn.defs), 3) + lambda*nn.W{k, nn.lstm.i});
            
            % Gradients of the recurrent connection weights
            dJdW{k, nn.lstm.h} = 1/m*sum(pagemultWrapper(nn.d{k+1}(:,:,3:T+1), nn.A{k+1}.v(:,:,2:T), nn.defs), 3);      
            
            % Biases
            dJdB{k} = 1/m*sum(sum(nn.d{k+1}, 3), 2);
        case nn.defs.TYPES.CONVOLUTIONAL
            [dJdW{k,1}, dJdB{k}] = cnnConvGradTemporal(nn.d{k+1}, nn.A{k}.v, nn.W{k}, nn, k, m);
            dJdW{k,1} = dJdW{k,1} + 1/m*nn.p.lambda*nn.W{k,1};
        case nn.defs.TYPES.AVERAGE_POOLING
            % No update required
        otherwise
        	error('Unknown layer type!')
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
