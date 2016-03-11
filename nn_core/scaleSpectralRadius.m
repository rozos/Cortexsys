function W = scaleSpectralRadius(W, r)
%scaleSpectralRadius Given a SQUARE matrix, adjust the spectral radius to
%be a specified value. This is useful for setting the largest eigenvalues
%of the hidden-to-hidden weights in a recurrent neural net. Setting the
%radius to ~1.1 is recommended to promote "rembering" while <1 promotes
%forgetting. If the radius is too large, the gradients will tend to
%explode.
% See: http://machinelearning.wustl.edu/mlpapers/papers/icml2013_sutskever13
% On the importance of initialization and momentum in deep learning

if (size(W,1) ~= size(W,2))
    error('scaleSpectralRadius(): input matrix must be square.');
end

% Determine the greatest eigenvalues
D = abs(eigs(W));
% Scale W so that largest eigenvalue is r
W = (r/max(D))*W;

end

