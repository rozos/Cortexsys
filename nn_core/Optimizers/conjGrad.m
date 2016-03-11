function nn = conjGrad(f, nn, defs, Xts, Yts, yts, y, plotTitle)
% CONJGRAD: Mini-batch Conjugate Gradient with Secant line search and
% Polak-Ribiere method.
% This optimization algorithm incorporates curvature information through
% the "Secant" forward finite difference.
% SEE: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
%      Shewchuk, Jonathan Richard. "An introduction to the conjugate gradient method without the agonizing pain." (1994).

% Setup some parameters for easy access
N_l = nn.N_l;
p = nn.p;
PrintIter = 1; 

% Debugging on?
if (defs.plotOn == true)
    Costs = precision(zeros(p.maxIter, 1), defs);
end
testError = []; 
trainError = []; 
    
% Load data onto GPU
nn.gpu(nn.W, nn.b);

if(defs.plotOn)
    figNum = 1000;
    figure(figNum); % Create plotting figure
end

if exist('OCTAVE_VERSION') == 0
  h.waitbar = waitbar(0,'1','Name','Training Network',...
              'CreateCancelBtn',...
              'setappdata(gcbf,''canceling'',1)');
  setappdata(h.waitbar,'canceling',0);
else
  h.waitbar = waitbar(0,'1','Name','Training Network');
  setappdata(h.waitbar,'canceling',0);
end

totalTime = 0; % Keep track of total running time
tic;

% Lowest cost achieves to date
Jbest = 1e10;
i = 0; % Main CG iteration
n = 0; % CG iterations since last reset

% If mini-batch is enabled, select a random sub-set of full training set
if (p.miniBatchSize ~= 0)
    mb = randperm(nn.m, p.miniBatchSize);
else
    mb = 1:nn.m;
end
% Compute initial CG residual/direction
x = packParams(nn.W, nn.b, nn);
[Jprev, dJdW, dJdB] = feval(f, nn, mb, true);
dJdx = packParams(dJdW, dJdB, nn);
r = -dJdx;
d = r; %initial CG direction (same as gradient)
while (i < p.maxIter)
    i = i+1;
    for q=1:p.cg.mbIters
        %%%%%%%%%%%%%%%%%%%%%%%% MAIN CG SECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
        % Compute forward directional gradient
        xd = x + p.cg.sigma0*d;
        unpackParams(xd, nn);
        [J_xd, dJdW_xd, dJdB_xd] = feval(f, nn, mb, false);
        dJdxd = packParams(dJdW_xd, dJdB_xd, nn);
        
        % Compute alpha which would minimize a parabolic approximation
        % of f(x) in the direction of d
        alpha0 = max(-p.cg.sigma0 * dJdx'*d / ((dJdxd - dJdx)'*d), 0);
        
        % Perform line search
        alpha = alpha0;
        j = 0;
        while (j < p.cg.jmax)
            xtmp = x + alpha*d;
            unpackParams(xtmp, nn);
            [J, dJdW, dJdB] = feval(f, nn, mb, false); % Evaluate cost at min of parabola
            % Uncheck the below to see how error progresses for a particular minibatch
            %disp([num2str(J) ', ' num2str(alpha) ', ' num2str(norm(d)), ', ' num2str(r'*d)]);
            
            % If cost went up, try a smaller step size and repeat
            if isnan(J) || (J > Jprev)
                disp('Overstepped.');
                if (j < p.cg.jmax-1)
                    alpha = 0.5*alpha;
                end
            else
                % Cost decreased, we are done
                break;
            end
            
            j = j+1; % Increment line search attempt number
        end
        x = x + alpha*d;
        Jprev = J;
        dJdx = packParams(dJdW, dJdB, nn);
        
        r_prev = r;
        r = -dJdx;
        beta = r'*(r-r_prev)/(r_prev'*r_prev);

        n = n+1; % Increment CG reset counter
        if (n == p.cg.N) || (beta <= 0) || (r'*d < 0)
           % Reset CG direction
           d = r;
           n = 0;
        else
           % Update CG direction
           d = r + beta*d; 
        end
    end
    
    %% Prepapre for new main CG iteration %%
    % Pick a new minibatch
    if (p.miniBatchSize ~= 0) && (i > 0)
        mb = randperm(nn.m, p.miniBatchSize);
    end
    
    % At start of new mini-batch, recompute cost/grad with new set of
    % random parameters
    [Jprev, dJdW, dJdB] = feval(f, nn, mb, true);
    dJdx = packParams(dJdW, dJdB, nn);
    % Reset CG
    r = -dJdx;
    d = r; %initial CG direction (same as gradient)
    n = 0; 
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if (J < Jbest)
        ibest = i;
        Jbest = J; 
        Wbest = nn.W;
        Bbest = nn.b;
    end

    if((mod(i,PrintIter)==0) || i==1)
        deltaT = toc/PrintIter;
        totalTime = totalTime + PrintIter*deltaT;
        avgTimePerIter = totalTime/i;
        waitbar(i/double(p.maxIter),h.waitbar,sprintf('Time remaining: %.2f minutes',(p.maxIter - i)*avgTimePerIter/60));
        
        % fprintf doesn't print on workers (w/ spmd)
        disp(sprintf('Iteration %4i | Cost: %4.3e | alpha: %4.3e | Time: %4.1f ms',...
                i, gather(J), alpha, 1e3*deltaT));

        % Determine test-set accuracy %
        % Update weights and biases to new values in main object
        if (~isempty(yts)) && ((mod(i,1*PrintIter)==0) || i==1)
            % Compute training set error
            [pred, ~]= predictDeep(nn.A{end}.v);
            trainError(end+1) = 100 - mean(double(pred == y(mb))) * 100;
            
            % Compute test set error
            % Swap in test set
            Ytmp = nn.Y;
            nn.A{1} = Xts;
            nn.Y = Yts;
            [pred, ~]= predictDeep(feedForward(nn, numel(mb), false, true));
            testError(end+1) = 100 - mean(double(pred == yts)) * 100;
            %Swap out test set
            nn.Y = Ytmp;
            
            disp(sprintf('  Test Set Error: %g%%, Training Set Error: %g%%', testError(end), trainError(end)));
        end
        if exist('OCTAVE_VERSION') > 0
            % write out disp() during loop execution if in Octave
            fflush(stdout);
        end
        tic
        
        if getappdata(h.waitbar,'canceling')
            disp('Training terminated early by user.')
            break
        end
    end

    if (defs.plotOn == true)
        if((mod(i,PrintIter)==0) || i==1)
            if ~isempty(yts)
            	plotTraining(Costs(1:i), figNum, plotTitle, trainError, testError);
            else
             	plotTraining(Costs(1:i), figNum, plotTitle);
            end
        end
    
        Costs(i) = gather(J);
    end
end
delete(h.waitbar);

% Present the best results achieved during the optimization loop
for k=1:N_l-1
    nn.W{k} = Wbest{k};
    nn.b{k} = Bbest{k};
end

disp(['Lowest cost on iteration ' num2str(ibest) '.']);

if (defs.plotOn == true)
    if ~isempty(yts)
    	plotTraining(Costs(1:i), figNum, plotTitle, trainError, testError);
    else
        plotTraining(Costs(1:i), figNum, plotTitle);
    end
end

end
