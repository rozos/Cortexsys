classdef sigmoid < handle
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
       type = 4;
       name = 'sigmoid';
    end

    methods
        function obj = sigmoid(varargin) 
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
            obj.alpha = precision(0.05, obj.defs); % gradient descent learning rate
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
            g = 1.0 ./ (1.0 + exp(-z));
        end
        
        function gp = grad(~,a)
           gp = a.*(1-a);
        end
        
        % Gradient at the output layer %
        function gp = ograd(obj,a)
            if (obj.costType == obj.defs.COSTS.LOGISTIC_REGRESSION)
                gp = 1; % logistic regression
            %elseif (obj.costType == obj.defs.COSTS.SQUARED_ERROR)    
            else
                gp = obj.grad(a); % squared error
            end
        end
        
        function J = cost(obj,Y, A, m, t, param)
            if (obj.costType == obj.defs.COSTS.LOGISTIC_REGRESSION)
                J = logisticCostFun(Y, A, m, t);
            elseif (obj.costType == obj.defs.COSTS.SQUARED_ERROR)
                J = squaredErrorCostFun(Y, A, m, t);
            elseif (obj.costType == obj.defs.COSTS.CUSTOM_ERROR)
                if exist( 'param', 'var' )
                    J = customErrorCostFun(Y, A, m, t, param);
                else
                    J = customErrorCostFun(Y, A, m, t );
                end
            end
        end
    end
end
