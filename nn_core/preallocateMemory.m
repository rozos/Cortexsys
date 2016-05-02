function preallocateMemory(nn, m, T)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    for k=2:nn.N_l
        switch nn.l.typ{k}
            case nn.defs.TYPES.FULLY_CONNECTED
                ztmp = zerosWrapper([nn.l.szo{k}(1), m, T+2], nn.defs);
                nn.d{k} = ztmp;
                nn.A{k}.v = ztmp;
            case nn.defs.TYPES.RECURRENT
                ztmp = zerosWrapper([nn.l.szo{k}(1), m, T+2], nn.defs);
                nn.d{k} = ztmp;
                nn.A{k}.v = ztmp;
            case nn.defs.TYPES.LSTM
                % LSTM internal activations
                ztmp = zerosWrapper([nn.l.szo{k}(1), m, T+2], nn.defs);
                nn.s_c{k} = ztmp; nn.eps_s{k} = ztmp; nn.eps_c{k} = ztmp;
                
                nn.d{k} = zerosWrapper([4*nn.l.szo{k}(1), m, T+2], nn.defs);
                
                % LSTM internal gate activations
                nn.A_lstm{k} = zerosWrapper([4*nn.l.szo{k}(1), m, T+2], nn.defs);
                
                % LSTM output activation
                nn.A{k}.v = zerosWrapper([nn.l.szo{k}(1), m, T+2], nn.defs);
            case {nn.defs.TYPES.CONVOLUTIONAL, nn.defs.TYPES.AVERAGE_POOLING}
                ztmp = zerosWrapper([nn.l.szo{k}(2), nn.l.szo{k}(3), nn.l.szo{k}(1), m, T+2], nn.defs);
                nn.d{k} = ztmp;
                nn.A{k}.v = ztmp;
        end
    end
end

