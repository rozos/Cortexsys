classdef LReLU < handle
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
       type = 8;
       name = 'LReLU';
    end

    methods
        function obj = LReLU(varargin) 
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
            obj.alpha = precision(0.01, obj.defs); % gradient descent learning rate
            obj.rho = precision(0.95, obj.defs); % AdaDelta hyperparameter (don't generally need to modify)
            obj.eps = precision(1e-6, obj.defs); % AdaDelta hyperparameter (don't generally need to modify)
        end
        
        function [W, b] = initializeWeightsBiases(obj, L_out, L_in, varargin)
            % A combination of basic algebra and empirical testing found
            % this to be a reasonable initialization that
            % eliminates/reduces stuck units and improves convergence
            % speed.
            if numel(varargin) == 1
                if strcmp(varargin{1}, 'lstm')
                    half_interval = 2*sqrt(1/(L_out+4*L_in));
                end
            else
                half_interval = 2*sqrt(1/(L_out+L_in));
            end
            W = half_interval*(2*rand(L_out, L_in, obj.defs.PRECISION) - 1);
            b = 0.15*precision(ones(L_out, 1), obj.defs);
        end
        
        function z = activ(~,z)
            I = find(z<=0);
            z(I) = 0.01*z(I);
        end
        
        function z = grad(~,z)
            %z(z>0) = 1;
            %z(z<=0) = 0.01;
            z = (z > 0) + 0.01 * (z <= 0);
        end
        
        % Gradient at the output layer %
        function gp = ograd(obj,z)
            %if (obj.costType == obj.defs.COSTS.SQUARED_ERROR)    
            %    gp = obj.grad(z); % squared error
            %end
            gp = obj.grad(z); % squared error
        end
        
        function J = cost(obj, Y, A, m, t, param)
            if (obj.costType == obj.defs.COSTS.CUSTOM_ERROR)    
                if exist( 'param', 'var' )
                    J = customErrorCostFun(Y, A, m, t, param);
                else
                    J = customErrorCostFun(Y, A, m, t );
                end
            else
                J = squaredErrorCostFun(Y, A, m, t);
            end
        end
    end
end
