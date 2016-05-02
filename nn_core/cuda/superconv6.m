function [Ao, p] = superconv6(A, d, p)
% SUPERCONV3 Perform convolution of a set of 3D "images" (4D array) with a
% set of 2D kernels (3D array), giving a 4D output. 
% For convolutional neural networks, the third dimension can be summed
% (e.g. sum(Ao,3)) and then the activation function can be applied.
% For best performance, store the parameters structure, p, which contains
% all of the setup to quickly invoke the convolution.

% Setup kernel and parameters and return them to caller for speed (reuse)
if isempty(p)
    tmp = A(1,1,1,1);
    if (isa(gather(tmp), 'single'))
        PRECISION = 'single';
    else
        PRECISION = 'double';
    end
    
    p.Nd =      int32(size(d)); % dimension of kernel
    p.Nd_even =  int32(not(mod([p.Nd(1) p.Nd(2)], 2)));
    p.Nd_odd =   int32(mod([p.Nd(1) p.Nd(2)], 2));
    p.Ndh =     ([p.Nd(1) p.Nd(2)] - p.Nd_odd)/2;
    p.Na =      int32(size(A));
    
    p.Nt = p.Na(5);
    p.Ni = p.Na(4);
    p.Nin = p.Na(3);
    p.Nout = p.Nd(3);
    
    % pre-allocate output memory
    p.result = gpuArray.zeros(p.Na(1)-p.Nd(1)+1, p.Na(2)-p.Nd(2)+1, p.Nin, p.Nout, p.Ni, p.Nt, PRECISION);
    
    % load and setup cuda kernel
    p.cudakernel = parallel.gpu.CUDAKernel('superconv6.ptx','superconv6.cu','superconv6');
    p.cudakernel.GridSize = [p.Na(1)-p.Nd(1)+1, p.Na(2)-p.Nd(2)+1, p.Ni*p.Nt];
    p.cudakernel.ThreadBlockSize = [p.Nin, p.Nout, 1];
    
    p.Nd = gpuArray(p.Nd);
    p.Nd_even = gpuArray(p.Nd_even);
    p.Nd_odd = gpuArray(p.Nd_odd);
    p.Ndh = gpuArray(p.Ndh);
    p.Na = gpuArray(p.Na);
    p.Ni = gpuArray(p.Ni);
    p.Nt = gpuArray(p.Nt);
end

Ao = feval(p.cudakernel, p.result, A, d, p.Na(1), p.Na(2), p.Ni, p.Nt, p.Nd(1),...
           p.Nd(2), p.Ndh(1), p.Ndh(2), p.Nd_even(1), p.Nd_even(2));

end

