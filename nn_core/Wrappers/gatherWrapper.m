function X = gatherWrapper(X, defs)
    if (defs.useGPU)
        if (iscell(X))
            if (isa(X{1}, 'varObj'))
                for n=1:numel(X)
                    X{n}.v = gather(X{n}.v);
                end
            else
                X = cellfun(@(x) gather(x), X, 'UniformOutput',false);
            end
        else
            X = gather(X);
        end
    end
end