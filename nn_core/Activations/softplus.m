classdef softplus < handle
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
       type = 3;
       name = 'softplus';
    end

    methods
        function obj = softplus(varargin) 
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
            obj.alpha = precision(0.025, obj.defs); % gradient descent learning rate
            obj.rho = precision(0.95, obj.defs); % AdaDelta hyperparameter (don't generally need to modify)
            obj.eps = precision(1e-6, obj.defs); % AdaDelta hyperparameter (don't generally need to modify)
        end
        
        function [W, b] = initializeWeightsBiases(obj, L_out, L_in)
            half_interval = 5*sqrt(1/(L_out+L_in));
            W = half_interval*(2*rand(L_out, L_in, obj.defs.PRECISION) - 1);
            b = 0.35*precision(ones(L_out, 1), obj.defs);
        end
        
        function g = activ(~,z)
            g = log(1+exp(z));
        end
        
        function gp = grad(~,a)
            % Note: gp = exp(z)./(1+exp(z)) = (exp(a)-1)./exp(a) when
            % solved for a.
          
            gp = (exp(a)-1)./exp(a);
        end
        
        % Gradient at the output layer %
        function gp = ograd(obj,a)
            if (obj.costType == obj.defs.COSTS.SQUARED_ERROR)    
                gp = obj.grad(a); % squared error
            end
        end
        
        function J = cost(~,Y, A, m, t)
            J = squaredErrorCostFun(Y, A, m, t);
        end
    end
end
