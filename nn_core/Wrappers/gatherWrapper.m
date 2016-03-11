function X = gatherWrapper(X, defs)
    if (defs.useGPU && (defs.numThreads > 1))
        X = gather(X);
    end
end