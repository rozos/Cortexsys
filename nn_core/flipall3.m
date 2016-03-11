% Flip 3D matrix along all three dimensions
function X=flipall3(X, dim)
        X = flip(rot90(X,2), dim);
end

