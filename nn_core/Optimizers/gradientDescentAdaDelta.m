% Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." arXiv preprint arXiv:1212.5701 (2012).
% AdaDelta is much less sensitive to hyperparameters and more numerically
% stable that gradient descent with momentum. However, performance is
% comparable.
function nn = gradientDescentAdaDelta(f, nn, defs, Xts, Yts, yts, y, plotTitle)
% Setup some parameters for easy access
N_l = nn.N_l;
p = nn.p;
PrintIter = 10; 

% AdaDelta hyperparameters
rho = p.rho;
eps = p.eps;

% AdaDelta RMS matricies
numParams = 0;
for k=1:N_l-1
    for j=nonEmptyCells(nn.W(k,:))
        E_dJdW{k,j} = zerosWrapper(size(nn.W{k,j}), defs);
        E_dW{k,j} = zerosWrapper(size(nn.W{k,j}), defs);
    end
    
    E_dJdB{k} = zerosWrapper(size(nn.b{k}), defs);
    E_dB{k} = zerosWrapper(size(nn.b{k}), defs);
end
dW = cell(N_l-1);
dB = cell(N_l-1);

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

i = 0;
totalTime = 0; % Keep track of total running time
tic;
% Set initial choice for the training batch
r = 1:nn.m;
Jbest = 1e10;
while (i < p.maxIter)
    i = i+1;

    % If mini-batch is enabled, select a random sub-set of full training set
    if (p.miniBatchSize ~= 0)
        r = randperm(nn.m, p.miniBatchSize);
    end
    
    % Compute gradients and cost (back-prop)
    [J, dJdW, dJdB] = feval(f, nn, r, true);
    
    % Compute AdaDelta RMS variables
    for k=1:N_l-1   
        if isempty(dJdW{k})
            % Skip pooling layers
            continue;
        end

        for j=nonEmptyCells(nn.W(k,:))
            % First, compute expectations of gradient time series
            E_dJdW{k,j} = rho*E_dJdW{k,j} + (1-rho)*dJdW{k,j}.^2;
            
            % Next, compute weight/bias updates from RMSs and gradient
            dW{k,j} = -sqrt((E_dW{k,j}+eps)./(E_dJdW{k,j}+eps)).*dJdW{k,j};
            
            % Then, update expectations of weight/bias updates (lags by one
            % time step over gradient expectations)
            E_dW{k,j} = rho*E_dW{k,j} + (1-rho)*dW{k,j}.^2;
            
            % Finally, update parameters
            nn.W{k,j} = nn.W{k,j} + dW{k,j};
        end
        
        % Do the same as above for the biases
        E_dJdB{k} = rho*E_dJdB{k} + (1-rho)*dJdB{k}.^2;
        dB{k} = -sqrt((E_dB{k}+eps)./(E_dJdB{k}+eps)).*dJdB{k};
        E_dB{k} = rho*E_dB{k} + (1-rho)*dB{k}.^2;
        nn.b{k} = nn.b{k} + dB{k};
    end

    if (J < Jbest)
        ibest = i;
        Jbest = gather(J); 
        Wbest = nn.W;
        Bbest = nn.b;
    end
   
    if((mod(i,PrintIter)==0) || i==1)
        deltaT = toc/PrintIter;
        totalTime = totalTime + PrintIter*deltaT;
        avgTimePerIter = totalTime/i;
        waitbar(i/double(p.maxIter),h.waitbar,sprintf('Time remaining: %.2f minutes',(p.maxIter - i)*avgTimePerIter/60));
        
        % fprintf doesn't print on workers (w/ spmd)
        disp(sprintf('Iteration %4i | Cost: %4.3e | Time: %4.1f ms',...
                i, gather(J), 1e3*deltaT));

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
            [pred, ~]= predictDeep(feedForward(nn, numel(r), false, true));
            testError(end+1) = 100 - mean(double(pred == yts)) * 100;
            %Swap out test set
            nn.Y = Ytmp;

            disp(sprintf('  Test Set Error: %g%%, Training Set Error: %g%%', testError(end), trainError(end)));
        end
        if exist('OCTAVE_VERSION', 'var') > 0
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
        if((mod(i,10*PrintIter)==0) || i==1)
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
