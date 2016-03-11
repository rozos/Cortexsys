function J = squaredErrorCostFun(Y, A, m, t)
    J = (Y.v(:,:,t)-A.v(:,:,t)).^2;
    J = 1/2/m*sum(J(:));
end