function unpackParams(P, nn)
    idx1 = 1;
    idx2 = 0;
    for k=1:nn.N_l-1   
            if (nn.l.typ{k} == nn.defs.TYPES.AVERAGE_POOLING)
                % Skip pooling layers
                continue;
            end

            for j=nonEmptyCells(nn.W(k,:))
                Nw = numel(nn.W{k,j});
                idx2 = idx2 + Nw;
                
                nn.W{k,j} = reshape(P(idx1:idx2), size(nn.W{k,j}));
                
                idx1 = idx1 + Nw;
            end 
            Nb = numel(nn.b{k});
            idx2 = idx2 + Nb;
            
            nn.b{k} = reshape(P(idx1:idx2), size(nn.b{k}));
            
            idx1 = idx1 + Nb;
    end
end