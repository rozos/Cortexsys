function [dJdW, dJdB] = cnnConvGrad(d, A, W, nn, k, m)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    if nn.defs.useGPU
        [dJdW, nn.cuda{k,3}] = superconv4(A, d, nn.cuda{k,3});
        dJdW = 1/m*sum(dJdW, 5);
    else
        dJdW = zerosWrapper(size(W), nn.defs);
        for j=1:nn.l.szo{k}(1) % loop over input maps
            for i=1:nn.l.szo{k+1}(1) % loop over output maps
                dJdW(:,:,j,i) = 1/m*convn((A(:,:,j,:)), flipall3(d(:,:,i,:), 4), 'valid');
            end
        end  
    end
    
    dJdB = 1/m*squeeze(sum(sum(sum(d,1),2),4));
end

