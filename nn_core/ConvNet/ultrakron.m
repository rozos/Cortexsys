function C = ultrakron(A,B)
% Compute kronecker product between 3D and 2D matricies in "pages"
% In other words, for each 2D slice of a 3D array, compute kron with B.

    sza = size(A);
    szb = size(B);
    maxLen = max(length(sza), length(szb));

    if length(sza) < maxLen
        sza = [sza ones(1,length(szb) - length(sza))];
    elseif length(szb) < maxLen
        szb = [szb ones(1, length(sza) - length(szb))]; 
    end

    % Multiply A and B
    C = reshape(A, numel(A), 1) * reshape(B, 1, numel(B));

    % Rearrange result
    C = reshape(C, [sza szb]);
    C = permute(C, reshape([maxLen+1:2*maxLen; 1:maxLen], 1, 2*maxLen));
    C = reshape(C, sza.*szb);
end
