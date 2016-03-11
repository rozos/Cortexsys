function V = ascii2onehot(str, vmap)  
    vocabSize = numel(vmap);

    vec = double(str);
    % Map all ASCII values to the reduced set
    for i=1:vocabSize
        vec(vec==vmap(i)) = i;
    end
    
    V = speye(vocabSize);
    V = V(vec,:)';
end