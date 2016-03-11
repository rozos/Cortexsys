% O(h^2) center difference formula
function [A0, dAdX] = computeNumericalActivationGradient(nn)
    h = 1e-4;
    Nx = size(nn.X.v, 1);
    
    % Save the original, non-replicated input data
    X0 = nn.X.v;
    
    % Expand input data into a square matrix
    % Perturb each input unit simultaneously to avoid looping
    nn.X.v = repmat(nn.X.v, 1,size(nn.X.v,1));
    nn.A{1} = nn.X; % Load the new input to the first layer
    h = diag(h*ones(1,Nx));

    A0 = feedForward(nn, Nx, false, true);

    nn.X.v = nn.X.v - h;
    A1 = feedForward(nn, Nx, false, true);

    nn.X.v = nn.X.v + 2*h;
    A2 = feedForward(nn, Nx, false, true);

    dAdX = (A2 - A1) / (2*h(1,1));

    % Restore original input data
    nn.X.v = X0;    
end