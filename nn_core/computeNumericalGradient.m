% O(h^2) center difference formula
function [dJdW, dJdB] = computeNumericalGradient(costFun,nn)
%COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
%and gives us a numerical estimate of the gradient.
%   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
%   gradient of the function J around theta. Calling y = J(theta) should
%   return the function value at theta.

% Notes: The following code implements numerical gradient checking, and 
%        returns the numerical gradient.It sets numgrad(i) to (a numerical 
%        approximation of) the partial derivative of J with respect to the 
%        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
%        be the (approximately) the partial derivative of J with respect 
%        to theta(i).)
%                

r = 1:nn.m;
h = gpuArrayWrapper(1e-5, nn.defs);
fprintf('Using O(h^2) center difference with h=%e\n', h);

% Loop over layers
for k=1:nn.N_l-1
    if (isempty(nn.b{k}))
        % skip pooling layers
        continue;
    end
    
    % Put W into a 4D shape if 2D
    if (ndims(nn.W{k}) == 2)
        wsz = size(nn.W{k});
        nn.W{k} = reshape(nn.W{k}, wsz(1), wsz(2), 1, 1);
    end
    
	% Loop over all elements in W
    if ((nn.l.typ{k+1} == nn.defs.TYPES.LSTM) || (nn.l.typ{k+1} == nn.defs.TYPES.RECURRENT)); E = 2; else E = 1; end
    for e=1:E
        for a=1:size(nn.W{k,e}, 1)
            for b=1:size(nn.W{k,e}, 2)
                for c=1:size(nn.W{k,e}, 3)
                    for d=1:size(nn.W{k,e}, 4)

                        nn.W{k,e}(a,b,c,d) = nn.W{k,e}(a,b,c,d) - h;
                        [J1, ~] = costFun(nn,r,false);

                        nn.W{k,e}(a,b,c,d) = nn.W{k,e}(a,b,c,d) + 2*h;
                        [J2, ~] = costFun(nn,r,false);
                        nn.W{k,e}(a,b,c,d) = nn.W{k,e}(a,b,c,d) - h; % restore back to original value

                        dJdW{k,e}(a,b,c,d) = (J2 - J1)/ (2*h);
                    end
                end
            end
        end
    end

    % Loop over all elements in W
    for p=1:size(nn.b{k},1)
        for q=1:size(nn.b{k},2)
            nn.b{k}(p,q) = nn.b{k}(p,q) - h;
            [J1, ~] = costFun(nn,r,false);
            
            nn.b{k}(p,q) = nn.b{k}(p,q) + 2*h;
            [J2, ~] = costFun(nn,r,false);
            nn.b{k}(p,q) = nn.b{k}(p,q) - h; % restore back to original value
            
            dJdB{k,1}(p,q) = (J2 - J1)/ (2*h);
        end
    end
end