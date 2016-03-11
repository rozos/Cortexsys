function S = rngWrapper(varargin)
S = [];

switch numel(varargin)
    case 1
        defs = varargin{1};
    case 2
        S = varargin{1};
        defs = varargin{2};
    otherwise
        error('rngWrapper(): Incorrent number of arguments');
end

if defs.useGPU && ~isempty(S)
    parallel.gpu.rng(S);
elseif defs.useGPU && isempty(S)
    S = parallel.gpu.rng('shuffle', 'Philox4x32-10');
elseif ~defs.useGPU && ~isempty(S)
    rng(S);
elseif ~defs.useGPU && isempty(S)
    S = rng('shuffle');
end
    