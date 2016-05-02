function [Ao, p] = superconv5(d, W, p)
% SUPERCONV3 Perform convolution of a set of 3D "images" (4D array) with a
% set of 2D kernels (3D array), giving a 4D output. 
% For convolutional neural networks, the third dimension can be summed
% (e.g. sum(Ao,3)) and then the activation function can be applied.
% For best performance, store the parameters structure, p, which contains
% all of the setup to quickly invoke the convolution.

% Setup kernel and parameters and return them to caller for speed (reuse)
if isempty(p)
    tmp = W(1,1,1,1);
    if (isa(gather(tmp), 'single'))
        PRECISION = 'single';
    else
        PRECISION = 'double';
    end
    
    p.Nw =      int32(size(W)); % dimension of kernel
    p.Nw_even =  int32(not(mod([p.Nw(1) p.Nw(2)], 2)));
    p.Nw_odd =   int32(mod([p.Nw(1) p.Nw(2)], 2));
    p.Nwh =     ([p.Nw(1) p.Nw(2)] - p.Nw_odd)/2;
    p.Nd =      int32(size(d));
    
    p.Ni = p.Nd(4);
    p.Nin = p.Nw(3);
    p.Nout = p.Nw(4);
    
    % pre-allocate output memory
    % Perform a FULL convolution (simulated zero padding)
    p.result = gpuArray.zeros(p.Nd(1)+p.Nw(1)-1, p.Nd(2)+p.Nw(2)-1, p.Nin, p.Ni, p.Nout, PRECISION);
    
    % load and setup cuda kernel
    p.cudakernel = parallel.gpu.CUDAKernel('superconv5.ptx','superconv5.cu','superconv5');
    p.cudakernel.GridSize = [p.Nd(1)+p.Nw(1)-1, p.Nd(2)+p.Nw(2)-1, p.Ni];
    p.cudakernel.ThreadBlockSize = [p.Nin, p.Nout, 1];
    
    p.Nw = gpuArray(p.Nw);
    p.Nw_even = gpuArray(p.Nw_even);
    p.Nw_odd = gpuArray(p.Nw_odd);
    p.Nwh = gpuArray(p.Nwh);
    p.Nd = gpuArray(p.Nd);
    p.Ni = gpuArray(p.Ni);
end

Ao = feval(p.cudakernel, p.result, d, W, p.Nd(1), p.Nd(2), p.Ni, p.Nw(1),...
           p.Nw(2), p.Nwh(1), p.Nwh(2), p.Nw_even(1), p.Nw_even(2));

end

