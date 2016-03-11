function A = cnnFlattenLayer(A, m)
%cnnFlattenLayer: "Flatten" the output of a convolutional or pooling layer
% such that it is a vector, like for a fully connected net. This is
% necessary when transitioning from a convolutional net to a fully
% connected net or recurrent net.
% force: flatten the layer, even if the next layer is convolutional
    
    if (ndims(A) == 4)
        A = reshape(A, [], m);
    end
end

