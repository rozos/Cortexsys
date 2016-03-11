classdef softmax < handle
    properties(GetAccess = 'public', SetAccess = 'private')
       costType;
       classID;
       alpha;
       rho;
       eps;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
       defs;
    end
    
    properties(GetAccess = 'public', Constant = true)
       type = 2;
       name = 'softmax';
    end

    methods
        function obj = softmax(varargin) 
            obj.classID = cputime;
            
            if (numel(varargin) == 3)
                obj.costType = varargin{2};
                obj.defs = varargin{1};
                obj.alpha = precision(varargin{3}, obj.defs);
            elseif (numel(varargin) == 2)
                obj.costType = varargin{2};
                obj.defs = varargin{1};
            elseif  (numel(varargin) == 1) 
                obj.defs = varargin{1};
            end  
            % Setup training hyperparameters
            obj.alpha = precision(0.1, obj.defs); % gradient descent learning rate
            obj.rho = precision(0.95, obj.defs); % AdaDelta hyperparameter (don't generally need to modify)
            obj.eps = precision(1e-6, obj.defs); % AdaDelta hyperparameter (don't generally need to modify)
        end
        
        function [W, b] = initializeWeightsBiases(obj, L_out, L_in)
            % Bengio, X. Glorot, Understanding the difficulty of training deep feedforward neuralnetworks, AISTATS 2010
            % http://deeplearning.net/tutorial/mlp.html
            half_interval = 4*sqrt(6/(L_out+L_in));
            W = half_interval*(2*rand(L_out, L_in, obj.defs.PRECISION) - 1);
            b = precision(zeros(L_out, 1), obj.defs);
        end
        
        function g = activ(~,z)
            arg1 = exp(z);
            arg2 = sum(arg1,1);
            g = bsxfun(@rdivide,arg1, arg2);
        end
        
        function gp = grad(~,~)
           disp('  **Softmax not defined for hidden layers!');
           gp = NaN;
        end
        
        % Gradient at the output layer %
        function gp = ograd(obj,~)
            if (obj.costType == obj.defs.COSTS.CROSS_ENTROPY)  
                gp = 1; % cross-entropy cost function
            end
        end
        
        function J = cost(~,Y, A, m, t)
            J = crossEntropyCostFun(Y, A, m, t);
        end
    end
end
