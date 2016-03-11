function [label, probs] = predictDeep(Aout)
    [pVal, label] = max(Aout', [], 2);
    probs = bsxfun(@rdivide, pVal, sum(Aout',2));
    
    label = gather(label);
    probs = gather(probs);
end
