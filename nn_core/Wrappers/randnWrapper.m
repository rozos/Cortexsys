function y = randnWrapper(sz, defs)
    
    if defs.useGPU
        y = gpuArray.randn(sz, defs.PRECISION);
    else
        y = randn(sz, defs.PRECISION);
    end
end