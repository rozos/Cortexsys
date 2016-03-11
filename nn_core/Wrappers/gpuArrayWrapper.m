function X = gpuArrayWrapper(X, defs)
    if (defs.useGPU == 1)
        if iscell(X)
            if ~issparse(X{1})
                X = cellfun(@(x) gpuArray(x), X, 'UniformOutput', false);
            end
        else
            X = gpuArray(X);
        end
    end
end