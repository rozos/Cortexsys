  classdef nnLayers < handle
    properties (Constant)
        SPECTRAL_RADIUS = 1.1;
    end
    properties(GetAccess = 'public', SetAccess = 'private')
       p; % parameters 
       l; % layers
       N_l; % number of layers
       m; % total number of training examples
       defs;
       classID;
       Nwb; % Total number of parameters (weights and biases) in network
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
       X; % input data
       W; % weights
       b; % biases
       Y; % Output layer targets
       A; % layer activations (input input)
       d; % backprop error terms
       Mdn; % Denoising mask
       Vdo; % Dropout mask
       cuda; % store cuda kernel stuff for performance optimization
       
       %%%% LSTM STUFF %%%
       lstm; % parameters for the LSTM
       rnn; % parameters for the RNN
       A_lstm; % LSTM gate activations (not including cell outputs)
       s_c; % LSTM cell values
       eps_s; % for LSTM cell backprop
       eps_c; % for LSTM cell output backprop
       sigmoid_af; % sigmoidal activation for LSTM gates
    end
    
    methods
        function obj = nnLayers(params, layers, X, Y, W, b, defs) 
           obj.classID = cputime;
           obj.p = params;
           obj.l = layers;
           obj.N_l = precision(numel(layers.af), defs); % number of layers in the network
           
           if iscell(X.v)
               switch ndims(X.v{1})
                   case {2,3}
                       % 2D input is data X training example
                        obj.m = precision(size(X.v{1}, 2), defs); % number of training examples
                   case {4,5}
                       % 4D 
                        obj.m = precision(size(X.v{1}, 4), defs); % number of training examples
               end
           else
               switch ndims(X.v)
                   case {2,3}
                       % 2D input is data X training example
                        obj.m = precision(size(X.v, 2), defs); % number of training examples
                   case {4,5}
                       % 4D 
                        obj.m = precision(size(X.v, 4), defs); % number of training examples
               end
           end
           
           obj.defs = defs;
           
           obj.X = X;
           obj.W = W;
           obj.b = b;
           obj.Y = Y;
           
           if defs.useGPU
               obj.cuda = cell(obj.N_l, 3);
           end
        end
        
        function disableCuda(obj)
            %obj.cuda = {};
            %obj.W = cellfun(@(x) gather(x), obj.W, 'UniformOutput',false);
            %obj.b = cellfun(@(x) gather(x), obj.b, 'UniformOutput',false);
            %obj.A = cellfun(@(x) gather(x), obj.A, 'UniformOutput',false);
            
            obj.W = gatherWrapper(obj.W, obj.defs);
            obj.b = gatherWrapper(obj.b, obj.defs);
            obj.A = gatherWrapper(obj.A, obj.defs);
            
            obj.defs.useGPU = false;
        end
        
        function enableCuda(obj)
            obj.defs.useGPU = true;
            %obj.cuda = {};
            %obj.W = cellfun(@(x) gpuArrayWrapper(x, obj.defs), obj.W, 'UniformOutput',false);
            %obj.b = cellfun(@(x) gpuArrayWrapper(x, obj.defs), obj.b, 'UniformOutput',false);
            %obj.A = cellfun(@(x) gpuArrayWrapper(x, obj.defs), obj.A, 'UniformOutput',false);
            
            obj.W = gpuArrayWrapper(obj.W, obj.defs);
            obj.b = gpuArrayWrapper(obj.b, obj.defs);
            obj.A = gpuArrayWrapper(obj.A, obj.defs);
        end
        
        function gpu(obj, varargin)
            if numel(varargin) == 2
                Win = varargin{1};
                bin = varargin{2};
            elseif numel(varargin) == 0
                Win = obj.W;
                bin = obj.b;
            else
                error('nnLayers.gpu(): improver number of arguments!');
            end
            
            obj.W = cellfun(@(x) gpuArrayWrapper(x, obj.defs), Win, 'UniformOutput',false);
            obj.b = cellfun(@(x) gpuArrayWrapper(x, obj.defs), bin, 'UniformOutput',false);
        end
        
        % Randomly initialize ALL layers
        function initWeightsBiases(obj)
            disp(['nnLayers(): randomly initializing all weights/biases for object ' num2str(obj.classID)]);
            
            % LSTM uses the columns of W to store input and hidden layer weights
            obj.W = cell(obj.N_l-1, 2);
            
            obj.Nwb = 0;
            
            for k=1:obj.N_l
                obj.A{k} = varObj({[]}, obj.defs); % Setup object for storing activations
                switch obj.l.typ{k}
                    case obj.defs.TYPES.INPUT
                         obj.A{1} = obj.X; % Setup object for storing activations
                         obj.l.szo{k} = obj.l.sz{k};
                    case obj.defs.TYPES.FULLY_CONNECTED
                         obj.l.szo{k} = obj.l.sz{k};
                         [obj.W{k-1,1}, obj.b{k-1}] = obj.l.af{k}.initializeWeightsBiases(obj.l.sz{k}(1), prod(obj.l.szo{k-1}));
                         obj.Nwb = obj.Nwb + numel(obj.W{k-1,1}) + numel(obj.b{k-1});
                    case obj.defs.TYPES.RECURRENT
                         obj.rnn.i = 1;
                         obj.rnn.h = 2;
                         
                         % Input to RNN layer weights
                         [obj.W{k-1,obj.rnn.i}, obj.b{k-1}] = obj.l.af{k}.initializeWeightsBiases(obj.l.sz{k}(1), prod(obj.l.szo{k-1}));
                         % Hidden to hidden (intra-layer) RNN weights
                         [obj.W{k-1,obj.rnn.h}, ~] = obj.l.af{k}.initializeWeightsBiases(obj.l.sz{k}(1), obj.l.sz{k}(1));
                         obj.W{k-1,obj.rnn.h} = scaleSpectralRadius(obj.W{k-1,obj.rnn.h}, obj.SPECTRAL_RADIUS);
                         
                         obj.l.szo{k} = obj.l.sz{k};
                         
                         obj.Nwb = obj.Nwb + numel(obj.W{k-1,obj.rnn.i}) + numel(obj.W{k-1,obj.rnn.h}) + numel(obj.b{k-1});
                    case obj.defs.TYPES.LSTM
                        % subscript legend: 
                        %    l: Input (control) gates
                        %    phi: Forget (control) gates
                        %    c: Cell input gates
                        %    ohm: output (control) gates
                         
                        % LSTMs have multiple weights per layer
                        % Indicies for LSTM weights, which are divided into
                        % two groups (input to LSTM and LSTM to LSTM, or
                        % hidden)
                        obj.lstm.i = 1; %column cell index of W
                        obj.lstm.h = 2; %column cell index of W
                        obj.lstm.phi = 1; % 3rd dim index of matrix in W
                        obj.lstm.c = 2; % 3rd dim index of matrix in W
                        obj.lstm.ohm = 3; % 3rd dim index of matrix in W
                        obj.lstm.l = 4; % 3rd dim index of matrix in W
                        
                        % NOTE: All LSTM gates must be sigmoid, except for the input and output of the cell.
                        obj.sigmoid_af = sigmoid(obj.defs, []);
                        % Input to LSTM layer
                        [obj.W{k-1,obj.lstm.i}, ~] = obj.sigmoid_af.initializeWeightsBiases(4*obj.l.sz{k}(1), prod(obj.l.szo{k-1}));
                        
                        % Recurrent connections (within hidden layer)
                        [obj.W{k-1,obj.lstm.h}, ~] = obj.sigmoid_af.initializeWeightsBiases(4*obj.l.sz{k}(1), obj.l.sz{k}(1));
                        
                        % Create bias vector
                        % Forget gate bias (phi) initialized to one to
                        % encourage rembering at beginning of training
                        obj.b{k-1} = zerosWrapper([4*obj.l.sz{k}(1), 1], obj.defs);
                        obj.b{k-1}(1+(obj.lstm.phi-1)*obj.l.sz{k}(1):obj.lstm.phi*obj.l.sz{k}(1)) = onesWrapper([obj.l.sz{k}(1), 1], obj.defs);
                        
                        % Store the indicies for the weight and output
                        % matrix coresponding to each LSTM gate
                        % This is useful since weights and outputs for each
                        % gate are packaged together in a single matrix
                        Nlstm = obj.l.sz{k}(1);
                        obj.lstm.idx_phi =  1+(obj.lstm.phi-1)*Nlstm    : obj.lstm.phi*Nlstm;
                        obj.lstm.idx_c =    1+(obj.lstm.c-1)*Nlstm      : obj.lstm.c*Nlstm;
                        obj.lstm.idx_ohm =  1+(obj.lstm.ohm-1)*Nlstm    : obj.lstm.ohm*Nlstm;
                        obj.lstm.idx_l =    1+(obj.lstm.l-1)*Nlstm      : obj.lstm.l*Nlstm;
                        
                        obj.l.szo{k} = obj.l.sz{k};
                        
                        obj.Nwb = obj.Nwb + numel(obj.W{k-1,obj.lstm.i}) + numel(obj.W{k-1,obj.lstm.h}) + numel(obj.b{k-1});
                    case obj.defs.TYPES.CONVOLUTIONAL                  
                        [Wtmp, ~] = obj.l.af{k}.initializeWeightsBiases(obj.l.sz{k}(1), obj.l.sz{k}(2)*obj.l.sz{k}(3)*obj.l.szo{k-1}(1));
                        obj.W{k-1,1} = reshape(Wtmp, [obj.l.sz{k}(2), obj.l.sz{k}(3), obj.l.szo{k-1}(1), obj.l.sz{k}(1)]);
                        
                        [~, obj.b{k-1}] = obj.l.af{k}.initializeWeightsBiases(obj.l.sz{k}(1), 1);
                        
                        % Compute output dimensions after convolution
                        obj.l.szo{k}(1) = obj.l.sz{k}(1);
                        obj.l.szo{k}(2) = obj.l.szo{k-1}(2) - obj.l.sz{k}(2) + 1;
                        obj.l.szo{k}(3) = obj.l.szo{k-1}(3) - obj.l.sz{k}(3) + 1;
                        
                        obj.Nwb = obj.Nwb + numel(obj.W{k-1,1}) + numel(obj.b{k-1});
                    case obj.defs.TYPES.AVERAGE_POOLING
                        % "W" for a pooling layer is just the pooling kernel. 
                        obj.W{k-1,1} = ones([obj.l.sz{k}(2), obj.l.sz{k}(3), obj.l.sz{k}(1)], obj.defs.PRECISION) / (obj.l.sz{k}(2)*obj.l.sz{k}(3));
                        obj.b{k-1} = [];
                        
                        % Compute output dimensions after pooling
                        obj.l.szo{k}(1) = obj.l.szo{k-1}(1);
                        obj.l.szo{k}(2) = obj.l.szo{k-1}(2)/obj.l.sz{k}(2);
                        obj.l.szo{k}(3) = obj.l.szo{k-1}(3)/obj.l.sz{k}(3);
                    otherwise
                        error('Unknown layer type!')
                end
            end
        end
    end
end