function J = logisticCostFun(Y, A, m, t)
    J = -Y.v(:,:,t).*log(A.v(:,:,t)) - (1-Y.v(:,:,t)).*log(1-A.v(:,:,t));
	J = 1/m*sum(J(:));
end