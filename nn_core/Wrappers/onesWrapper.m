function y = onesWrapper(sz, defs)
    
    if defs.useGPU
        y = gpuArray.ones(sz, defs.PRECISION);
    else
        y = ones(sz, defs.PRECISION);
    end
end