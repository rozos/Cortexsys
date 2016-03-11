function Aout = cnnConvolve(A, W, b, nn, k, m)
    numFilters = nn.l.szo{k}(1);
    
    % Pre-allocate output activations
    Aout = zerosWrapper([nn.l.szo{k}(2), nn.l.szo{k}(3), numFilters, m], nn.defs);
    
    if nn.defs.useGPU
        for j = 1:numFilters
            [Ac, nn.cuda{k,1}] = superconv3(A, W(:,:,:,j), nn.cuda{k,1});
            Aout(:, :, j, :) = sum(Ac, 3) + b(j);
        end       
    else
        for j = 1:numFilters
            Aout(:, :, j, :) = convn(A, flipall3(W(:,:,:,j), 3), 'valid') + b(j);
        end 
    end
    
    Aout = nn.l.af{k}.activ(Aout);
end

