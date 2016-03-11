clear
addpath('../');
gDev = gpuDevice(1);
system('rm *.ptx','-echo');
PRECISION = 'double';
Nbench = 1;

disp('Building, testing and benchmarking NVIDIA/CUDA accelerated convolutional net kernels.');
disp('  -> NVIDIA Cuda Toolkit 7.0 and either GCC or Visual Studio 2013 compilers must also be installed <'); disp(' ');

%%%%%%%%%%%%%%%%%%%%%%%%%%% SUPERCONV5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%{
disp('1. Compiling superconv5 (used in cnnUnconvolve)');
[status,result] = system('nvcc --compiler-bindir /usr/bin/gcc-4.9 -ptx superconv5.cu','-echo');
if status == 0
    disp('Running error test');
    Ni = int32(16); % number of images
    Nout = 3;
    Nd = int32([23 23 Nout Ni]); % dimension of kernel
    Nw = int32([5 6 4 Nout]); % dimension of image
    
    d = gpuArray.randn(Nd, PRECISION);
    W = gpuArray.randn(Nw, PRECISION);
    
    % test output for accuracy
    p = [];
    [out, p] = superconv5(d, W, p);
    out = sum(out, 5);
   
    for j=1:Nw(3) % loop over input maps
        dtmp = zeros([Nd(1)+Nw(1)-1, Nd(2)+Nw(2)-1, 1, Ni]);
        for i=1:Nw(4) % loop over output maps
            dtmp = dtmp + convn(d(:,:,i,:), rot90(W(:,:,j,i),2), 'full');
        end
        out_conv(:,:,j,:) = dtmp;
    end
    
    err = (out_conv - out).^2;
    err = sum(err(:));
    disp(['Numerical error is ' num2str(err) '.']);
    if err > 1e-16
        error('superconv5 failed numerical accuracy check!');
    end
    
    % benchmark
    disp('Running benchmarks');
    tic 
    for i=coder.unroll(1:Nbench, true)
        [out, p] = superconv5(d, W, p);
        out = sum(out, 5);
    end
    wait(gDev);
    tgpu = toc;
    disp([num2str(tgpu/Nbench*1e3) ' ms for kernel']);
    
    %d = gather(d);
    %W = gather(W);
    tic
    for i=coder.unroll(1:Nbench, true)
        for j=1:Nw(3) % loop over input maps
            dtmp = zeros([Nd(1)+Nw(1)-1, Nd(2)+Nw(2)-1, 1, Ni]);
            for k=1:Nw(4) % loop over output maps
                dtmp = dtmp + convn(d(:,:,k,:), W(:,:,j,k), 'full');
            end
            out_conv(:,:,j,:) = dtmp;
        end
    end
    wait(gDev);
    tcpu = toc;
    disp([num2str(tcpu/Nbench*1e3) ' ms for convn (matlab)']);
    disp(['Speed up is: ' num2str(tcpu/tgpu) 'X for superconv5']);
else
    error('Error compiling superconv5!');
end
clear out_conv;
disp('-----------------'); disp(' ');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% SUPERCONV4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%{
disp('2. Compiling superconv4 (used in cnnConvGrad)');
[status,result] = system('nvcc --compiler-bindir /usr/bin/gcc-4.9 -ptx superconv4.cu','-echo');
if status == 0
    disp('Running error test');
    Ni = int32(16); % number of images
    Nd = int32([9 6 5 Ni]); % dimension of kernel
    Na = int32([28 28 7 Ni]); % dimension of image
    
    A = gpuArray.randn(Na, PRECISION);
    d = gpuArray.randn(Nd, PRECISION);
    
    % test output for accuracy
    p = [];
    [out, p] = superconv4(A, d, p);
    out = sum(out, 5);
    
    for j=1:Na(3) % loop over input maps
        for i=1:Nd(3) % loop over output maps
            out_conv(:,:,j,i) = convn((A(:,:,j,:)), flipall3(d(:,:,i,:), 4), 'valid');
        end
    end 
    
    err = (out_conv - out).^2;
    err = sum(err(:));
    disp(['Numerical error is ' num2str(err) '.']);
    if err > 1e-16
        error('superconv4 failed numerical accuracy check!');
    end
    
    % benchmark
    disp('Running benchmarks');
    tic 
    for i=coder.unroll(1:Nbench, true)
        [out, p] = superconv4(A, d, p);
        out = sum(out, 5);
    end
    wait(gDev);
    tgpu = toc;
    disp([num2str(tgpu/Nbench*1e3) ' ms for kernel']);
    
    %d = gather(flipall3(d,3));
    %A = gather(A);
    tic
    for i=coder.unroll(1:Nbench, true)
        for j=1:Na(3) % loop over input maps
            for k=1:Nd(3) % loop over output maps
                out_conv(:,:,j,k) = convn((A(:,:,j,:)), flipall3(d(:,:,k,:), 4), 'valid');
            end
        end 
    end
    wait(gDev);
    tcpu = toc;
    disp([num2str(tcpu/Nbench*1e3) ' ms for convn (matlab)']);
    disp(['Speed up is: ' num2str(tcpu/tgpu) 'X for superconv4']);
else
    error('Error compiling superconv4!');
end
clear out_conv;
disp('-----------------'); disp(' ');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% SUPERCONV3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%{
disp('3. Compiling superconv3 (used in cnnConvolve)');
[status,result] = system('nvcc --compiler-bindir /usr/bin/gcc-4.9 -ptx superconv3.cu','-echo');
if status == 0
    disp('Running error test');
    Nk = int32([6 5 10]); % dimension of kernel
    Nm = int32([27 28]); % dimension of image
    Ni = int32(16); % number of images
    maps = gpuArray.randn(Nm(1), Nm(2), Nk(3), Ni, PRECISION);
    K = gpuArray.randn(Nk, PRECISION);
    
    % test output for accuracy
    p = [];
    [out, p] = superconv3(maps, K, p);
    out = sum(out, 3);
    
    out_conv = gather(convn(maps, flipall3(K,3), 'valid'));
    err = (out_conv - out).^2;
    err = sum(err(:));
    disp(['Numerical error is ' num2str(err) '.']);
    if err > 1e-16
        error('superconv3 failed numerical accuracy check!');
    end
    
    % benchmark
    disp('Running benchmarks');
    tic 
    for i=coder.unroll(1:Nbench, true)
        [x, p] = superconv3(maps, K, p);
        x = sum(x,3);
    end
    wait(gDev);
    tgpu = toc;
    disp([num2str(tgpu/Nbench*1e3) ' ms for kernel']);
    
    %K = gather(flipall3(K,3));
    %maps = gather(maps);
    K = flipall3(K,3);
    tic
    x = convn(maps, K, 'valid');
    for i=coder.unroll(1:Nbench, true)
        x = convn(maps, K, 'valid');
    end
    wait(gDev);
    tcpu = toc;
    disp([num2str(tcpu/Nbench*1e3) ' ms for convn (matlab)']);
    disp(['Speed up is: ' num2str(tcpu/tgpu) 'X for superconv3']);
else
    error('Error compiling superconv3!');
end
disp('-----------------'); disp(' ');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(' -> All kernels compiled and tested successfully! <-');
