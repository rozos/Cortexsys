function A = cnnUnflattenLayer(A, szo)
%cnnFlattenLayer: Undo cnnFlattenLayer()
    
    if (ndims(A) == 2)
        A = reshape(A, szo(2), szo(3), szo(1), []);
    end
end

