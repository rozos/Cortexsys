% (input feature maps, pooling kernel, proceeding layer output size, pooling layer output size, pooling layer size/dimensions)
function Aout = cnnPool(A, W, szi, szo, szp, nn, k, m)
    numFilters = szo(1);

    XpoolIdx = 1 : szp(2) : (szi(2) - szp(2) + 1);
    YpoolIdx = 1 : szp(3) : (szi(3) - szp(3) + 1);
    if nn.defs.useGPU
            [Ap, nn.cuda{k,1}] = superconv3(A, W, nn.cuda{k,1});
            Aout = Ap(XpoolIdx, YpoolIdx,:,:);
    else
        % Pre-allocate output activations
        Aout = zerosWrapper([szo(2), szo(3), numFilters, m], nn.defs);
    
        for j = 1:numFilters  
            % sum up all pixels within each patch
            Ap = convn(A(:, :, j, :), W(:,:,1), 'valid');
            Aout(:, :, j, :) = Ap(XpoolIdx, YpoolIdx,:,:);
        end
    end
end

