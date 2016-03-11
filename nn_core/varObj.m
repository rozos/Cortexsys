% varObj: The purpose of this object is two fold:
%         (1) To act as container for data that simplifies access
%         (2) Service as a "handle object" containing data. This permits
%         Matlab to treat the data as a pointer, reducing the need for
%         copying (performance speed up).

classdef varObj < handle
    properties(GetAccess = 'public', SetAccess = 'public')
       v;
       classID;
       name = 'varObj';
       defs;
       type; % Input data? If so, translate time axis
    end 

    methods
        function obj = varObj(v, defs, varargin) 
            obj.classID = cputime;
            obj.defs = defs;
            
            if numel(varargin) == 1
                obj.type = varargin{1};
            else
                obj.type = -1;
            end
            
            % If we are passed a varObj, extract the data instead of
            % building a nested object.
            if isobject(v) && ~isa(v, 'gpuArray')
                if strcmp(v.name, 'varObj')
                    obj.v = v.v;
                end
            else
                obj.v = v;
            end
        end
        
        % Extract a mini-batch from the input/output data
        function v = getmb(obj, r)
            if ((obj.type == obj.defs.TYPES.INPUT) || (obj.type == obj.defs.TYPES.OUTPUT));
                if iscell(obj.v)
                   % time series data where each time point is contained in a unique cell
                   v = cellfun(@(C) full(C(:,r)), obj.v, 'UniformOutput', false);
                   v = gpuArrayWrapper(precision(cat(3,v{:}),obj.defs), obj.defs);
                else
                   if (ndims(obj.v) == 2) % standard fully connected net data
                        v = gpuArrayWrapper(precision(full(obj.v(:,r)), obj.defs), obj.defs); 
                   elseif (ndims(obj.v) == 4) % convolutional net data
                        v = gpuArrayWrapper(precision(full(obj.v(:,:,:,r)), obj.defs), obj.defs); 
                   else
                       error(['nnLayers::getmb(), data has unsupported dimensionality of ' num2str(ndims(obj.v))]);
                   end
                end
            else
                error('varObj.getmb() only valid for input or output data');
            end
        end
    end
end
