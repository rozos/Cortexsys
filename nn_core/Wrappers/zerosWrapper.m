function y = zerosWrapper(sz, defs)
    
    if defs.useGPU
        y = gpuArray.zeros(sz, defs.PRECISION);
    else
        y = zeros(sz, defs.PRECISION);
    end
end