function A = cnnFlattenLayer(A, m, varargin)
%cnnFlattenLayer: "Flatten" the output of a convolutional or pooling layer
% such that it is a vector, like for a fully connected net. This is
% necessary when transitioning from a convolutional net to a fully
% connected net or recurrent net.
% force: flatten the layer, even if the next layer is convolutional

% varargin{1} is the time index, which is used when A has a time dimension
% (ndims == 3 or 5)

Narg = numel(varargin);

switch ndims(A)
    case 3
        if Narg == 1
            A = A(:,:,varargin{1});
        end
    case 4
        A = reshape(A, [], m);
    case 5
        if Narg == 1
            A = reshape(A(:,:,:,:,varargin{1}), [], m);
        else
            A = reshape(A, [], m, size(A,5));
        end
end
    
end

