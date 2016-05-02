function X = gpuArrayWrapper(X, defs)
    if (defs.useGPU == 1)
        if iscell(X)
            if (isa(X{1}, 'varObj'))
                if ~issparse(X{1}.v)
                    for n=1:numel(X)
                        X{n}.v = gpuArray(X{n}.v);
                    end
                end
            else
                if ~issparse(X{1})
                    X = cellfun(@(x) gpuArray(x), X, 'UniformOutput', false);
                end
            end
        else
            X = gpuArray(X);
        end
    end
end