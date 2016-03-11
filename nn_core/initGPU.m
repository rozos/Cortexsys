function GPU = initGPU(whichGPU)
    if (~isempty(whichGPU))
       disp(['Detected ' num2str(gpuDeviceCount()) ' GPUs.']); 
       GPU = gpuDevice(whichGPU);
       reset(GPU);
       disp(['Using GPU device ' GPU.Name ' with ' num2str(GPU.AvailableMemory/1024^3, 3) ' GB free.']);
    else
        GPU = [];
    end
end