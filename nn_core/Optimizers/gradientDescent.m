function nn = gradientDescent(f, nn, defs, Xts, Yts, yts, y, plotTitle)
% Setup some parameters for easy access
N_l = nn.N_l;
p = nn.p;
PrintIter = 10; 

% Debugging on?
if (defs.plotOn == true)
    Costs = precision(zeros(p.maxIter, 1), defs);
    Alphas = precision(zeros(p.maxIter, 1), defs);
end
testError = []; 
trainError = []; 
    
% Load data onto GPU
nn.gpu(nn.W, nn.b);

% Initialize cell arrays used during gradient descent
for k=1:N_l-1
    for j=nonEmptyCells(nn.W(k,:))
        dWp{k,j} = zerosWrapper(size(nn.W{k,j}), defs);
    end
    dbp{k} = zerosWrapper(size(nn.b{k}), defs);
    
    dW = cell(N_l-1, 1);
    db = cell(N_l-1, 1);
end

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

% Main gradient descent loop
alpha = p.alpha; % Learning rate as a function of iteration number
i = 0;
totalTime = 0; % Keep track of total running time
tic;

% Set initial choice for the training batch
r = 1:nn.m;
Jbest = 1e10;
while (i < p.maxIter)
    i = i+1;
    
    % If mini-batch is enabled, select a random sub-set of full training set
    %rng(seed);
    if (p.miniBatchSize ~= 0)
        r = randperm(nn.m, p.miniBatchSize);
    end
    
    % Compute gradients and cost (back-prop)
    [J, dJdWc, dJdBc] = feval(f, nn, r, true);
    
    % Compute actual weight and bias updates
    for k=1:N_l-1   
        if isempty(dJdWc{k})
            % Skip pooling layers
            continue;
        end
        
        for j=nonEmptyCells(nn.W(k,:))
            dW{k,j} = -alpha*dJdWc{k,j} + p.momentum*dWp{k,j};
            nn.W{k,j} = nn.W{k,j} + dW{k,j};
        end
        
        db{k} = -alpha*dJdBc{k} + p.momentum*dbp{k};
        nn.b{k} = nn.b{k} + db{k};     
    end

    % Store previous weight updates for use in momentum
    dWp = dW;
    dbp = db;

   %% Max-norm regularization
    % Helps prevent weights from blowing up when using large learning rates
    % Dropout paper does not recommend max-norm of output layer weights
    if (p.maxnorm ~= 0)
        for k=1:N_l-1
            % Compute norms of all columns for each weight matrix
            Wnorm{k} = sqrt(sum(nn.W{k}.^2,2));
            Wnorm{k}(Wnorm{k} <= p.maxnorm) = 1; % Set all norms not meeting threshold to 1
            nn.W{k} = bsxfun(@rdivide, nn.W{k}, Wnorm{k});
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
        if (~isempty(yts)) && ((mod(i,5*PrintIter)==0) || i==1)
            % Compute training set error
            [pred, ~]= predictDeep(nn.A{end}.v);
            trainError(end+1) = 100 - mean(double(pred == y(r))) * 100;
            
            % Compute test set error
            % Swap in test set
            Ytmp = nn.Y;
            nn.A{1} = Xts;
            nn.Y = Yts;
            nn.disableCuda();
            [pred, ~]= predictDeep(feedForward(nn, numel(yts), false, true));
            testError(end+1) = 100 - mean(double(pred == yts)) * 100;
            %Swap out test set
            nn.enableCuda();
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

    % Update learning rate
    alpha = p.alpha*p.alphaTau/(p.alphaTau+i);

    if (defs.plotOn == true)
        if((mod(i,10*PrintIter)==0) || i==1)
            if ~isempty(yts)
                    plotTraining(Costs(1:i), figNum, plotTitle, trainError, testError);
                else
                    plotTraining(Costs(1:i), figNum, plotTitle);
            end
        end
    
        Costs(i) = gather(J);
        Alphas(i) = alpha;
    end
end
delete(h.waitbar);

% Present the best results achieved during the optimization loop
Jbest = gather(Jbest);
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
