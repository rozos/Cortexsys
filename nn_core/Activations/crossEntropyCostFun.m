function J = crossEntropyCostFun(Y, A, m, t)
    J = Y.v(:,:,t).*log(A.v(:,:,t));
    J = -1/m*sum(J(:));
end