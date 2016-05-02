function [dJdW, dJdB] = cnnConvGradTemporal(d, A, W, nn, k, m)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    if nn.defs.useGPU
        [dJdW, nn.cuda{k,3}] = superconv6(A, d, nn.cuda{k,3});
        dJdW = 1/m*sum(sum(dJdW, 6),5);
        dJdB = 1/m*squeeze(sum(sum(sum(sum(d,1),2),4),5));
    else
        dJdW = zerosWrapper(size(W), nn.defs);
        dJdWt = dJdW;
        for t = 2:size(A,5)-1
            for j=1:nn.l.szo{k}(1) % loop over input maps
                for i=1:nn.l.szo{k+1}(1) % loop over output maps
                    dJdWt(:,:,j,i) = 1/m*convn(A(:,:,j,:,t), flipall3(d(:,:,i,:,t), 4), 'valid');
                end
            end
            dJdW = dJdW + dJdWt;
        end
        dJdB = 1/m*squeeze(sum(sum(sum(sum(d,1),2),4),5));
    end
end

