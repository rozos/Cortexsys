function c = onehot2ascii(vec, vmap)  
    vec = gather(vec);
    idx = find(vec==1);
    c = vmap(idx); %matlab indexing starts at one...
end