function [P] = packParams(W, b, nn)

    % Pre-allocate memory for parameters
    P = zerosWrapper([nn.Nwb 1], nn.defs);

    idx1 = 1;
    idx2 = 0;
    for k=1:nn.N_l-1   
            if (nn.l.typ{k} == nn.defs.TYPES.AVERAGE_POOLING)
                % Skip pooling layers
                continue;
            end

            for j=nonEmptyCells(W(k,:))
                Nw = numel(W{k,j});
                idx2 = idx2 + Nw;
                
                P(idx1:idx2) = W{k,j}(:);
                
                idx1 = idx1 + Nw;
            end 
            Nb = numel(nn.b{k});
            idx2 = idx2 + Nb;
            
            P(idx1:idx2) = b{k}(:);
            
            idx1 = idx1 + Nb;
    end
end

% Code to verify correctness of unpackParams and packParams routines:
%{
nn.initWeightsBiases();
Wtmp = nn.W;
btmp = nn.b;
W = packParams(nn.W, nn.b, nn.W, nn.b, nn); unpackParams(W, nn);
for k=1:nn.N_l-1   
        if isempty(nn.W{k})
            % Skip pooling layers
            continue;
        end

        for j=nonEmptyCells(nn.W(k,:))
            if ~isempty(find(Wtmp{k,j} ~= nn.W{k,j}))
                error('W Params dont match!');
            end
        end 
        if ~isempty(find(btmp{k} ~= nn.b{k}))
            error('Bias Params dont match!');
        end
end
%}