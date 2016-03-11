function [dA] = pagemultWrapper(d, A, defs)
% Given a 3D array (third dimension is time, or pages), multiply each 2D
% slice in parallel. Equivalent to for loop over d(:,:,t)*A(:,:,t). 
% Also, automatically transposes A!

if defs.useGPU
    dA = pagefun(@mtimes, d, pagefun(@transpose, A));
else
    dA = mmx('mult', d, A, 'nt');
end
end

