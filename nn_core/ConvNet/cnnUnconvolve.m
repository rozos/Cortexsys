function dout = cnnUnconvolve(d2, W, nn, k, m)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    szo = nn.l.szo{k};
    szo2 = nn.l.szo{k};
    szo3 = nn.l.szo{k+1};
    
    if nn.defs.useGPU
        [dout, nn.cuda{k,2}] = superconv5(d2, rot90(W,2), nn.cuda{k,2});
        dout = sum(dout, 5);
    else
        dout = zerosWrapper([szo2(2), szo2(3), szo2(1), m], nn.defs);
        
        for j=1:szo(1) % loop over input maps
            dtmp = zerosWrapper([szo2(2), szo2(3), 1, m], nn.defs);
            for i=1:szo3(1) % loop over output maps
                dtmp = dtmp + convn(d2(:,:,i,:), W(:,:,j,i), 'full');
            end
            dout(:,:,j,:) = dtmp;
        end        
    end
end

