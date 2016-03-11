function y = randWrapper(sz, defs)
    
    if defs.useGPU
        y = gpuArray.rand(sz, defs.PRECISION);
    else
        y = rand(sz, defs.PRECISION);
    end
end