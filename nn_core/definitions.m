classdef definitions < handle
    properties(GetAccess = 'public', SetAccess = 'public')
        useGPU = false;
    end
     
    properties(GetAccess = 'public', SetAccess = 'private')
       COSTS;
       TYPES;
       VERSION = 3.1;
       GPUs = {};
       Threads;
       numThreads;
       PRECISION;
       plotOn;
       classID;
       licText = ['\nCopyright (c) 2016 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 \n' ...
                'with Sandia Corporation, the U.S. Government retains certain rights in this software.\n\n' ...
                'Redistribution and use in source and binary forms, with or without modification,\n' ... 
                'are permitted provided that the following conditions are met:\n\n' ...
                    'Redistributions of source code must retain the above copyright notice, this list\n' ... 
                    'of conditions and the following disclaimer. Redistributions in binary form must \n' ...
                    'reproduce the above copyright notice, this list of conditions and the following \n' ...
                    'disclaimer in the documentation and/or other materials provided with the distribution.\n' ...
                    'Neither the name of the Sandia Corporation nor the names of its contributors may \n' ...
                    'be used to endorse or promote products derived from this software without specific \n' ...
                    'prior written permission.\n\n',...
                'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY \n' ...
                'EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES \n' ...
                'OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT \n' ...
                'SHALL SANDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, \n' ...
                'EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF \n' ...
                'SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS \n' ...
                'INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, \n' ...
                'STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY \n' ...
                'OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n'];
    end
    
    methods
        function obj = definitions(PRECISION, useGPU, whichThreads, plotOn) 
            obj.classID = cputime;
            
            obj.plotOn = plotOn;
            obj.PRECISION = PRECISION;
            
            obj.COSTS = struct();
            obj.COSTS.LOGISTIC_REGRESSION = 1;
            obj.COSTS.SQUARED_ERROR = 2;
            obj.COSTS.CROSS_ENTROPY = 3;
            
            obj.TYPES = struct();
            obj.TYPES.INPUT = 0;
            obj.TYPES.OUTPUT = 6;
            obj.TYPES.FULLY_CONNECTED = 1;
            obj.TYPES.CONVOLUTIONAL = 2;
            obj.TYPES.AVERAGE_POOLING = 3;
            obj.TYPES.LSTM = 4;
            obj.TYPES.RECURRENT = 5;
            
            obj.useGPU = useGPU;
            obj.Threads = whichThreads;
            obj.numThreads = numel(whichThreads);
            
            if exist('gather') == 0
              % Add gather dummy function to path if no Parallel Toolbox is installed
              addpath('../octave_wrappers');
              disp('Loaded gather() wrapper (no Parallel Computing Toolbox found).');
            end
            
            if useGPU
                % Initialize the GPUs
                if (length(whichThreads) <= gpuDeviceCount)
                    for i=1:length(whichThreads)
                        obj.GPUs{i} = initGPU(whichThreads(i));
                        if(~isempty(obj.GPUs{i}))
                            obj.useGPU = true;
                        else 
                            disp('Error initializing a GPU!');
                        end
                    end
                else
                    disp('Too many GPUs specified!');
                    obj.useGPU = false;
                end
            end
            
            % Display version, license and copyright information
            disp(['*** Cortexsys version ' num2str(obj.VERSION) ' initialized ***']);
            disp(['    Released under BSD License']);
            disp(['    Author: Jonathan A. Cox (jacox@sandia.gov, joncox@alum.mit.edu)']);
            disp(['    ->Using ' obj.PRECISION ' precision.']);
            
            if (exist('cortexsys_lic.mat', 'file') == 0)
               disp(sprintf(obj.licText));
            end
            
            % Write out file indicating that copyright and license were
            % shown
            licShown = true;
            save('cortexsys_lic.mat', 'licShown');
        end
    end
end