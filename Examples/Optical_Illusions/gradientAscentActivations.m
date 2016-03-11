function [Abest, Xbest] = gradientAscentActivations(nn, Ni, defs, plotTitle)
% Setup some parameters for easy access
N_l = nn.N_l;
p = nn.p;
PrintIter = 10; 

% Debugging on?
if (defs.plotOn == true)
    Costs = precision(zeros(p.maxIter, 1), defs);
    Alphas = precision(zeros(p.maxIter, 1), defs);
end

Abest = -1e10;

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

dXp = 0;

nn.gpu();

% Main gradient descent loop
alpha = p.alpha; % Learning rate as a function of iteration number
i = 0;
totalTime = 0; % Keep track of total running time
tic;
while (i < p.maxIter)
    i = i+1;

    nn.X.v = 7*nn.X.v/norm(nn.X.v);
    %nn.X.v = scaleRange(nn.X.v, [0 1]);
    [A, dAdX] = computeNumericalActivationGradient(nn);
    A = A(Ni);
    dAdX = dAdX(Ni,:);
    
    % Compute actual weight and bias updates
    dX = alpha*dAdX(:) + p.momentum*dXp;
    nn.X.v = nn.X.v + dX;
    
    % Store previous weight updates for use in momentum
    dXp = dX;
    
    if (A > Abest)
        ibest = i;
        Abest = A; 
        Xbest = nn.X.v;
    end

    if (A > 0.999)
        break;
    end
    
    if((mod(i,PrintIter)==0) || i==1)
        deltaT = toc/PrintIter;
        totalTime = totalTime + PrintIter*deltaT;
        avgTimePerIter = totalTime/i;
        waitbar(i/double(p.maxIter),h.waitbar,sprintf('Time remaining: %.2f minutes',(p.maxIter - i)*avgTimePerIter/60));
        
        % fprintf doesn't print on workers (w/ spmd)
        disp(sprintf('Iteration %4i | Cost: %4.3e | alpha: %4.3e | Time: %4.1f ms',...
                i, gather(A), alpha, 1e3*deltaT));

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
        if((mod(i,5*PrintIter)==0) || i==1)
            plotTraining(Costs(1:i), figNum, plotTitle);
        end
    
        Costs(i) = gather(A);
        Alphas(i) = alpha;
    end
end
delete(h.waitbar);

disp(['Highest activation on iteration ' num2str(ibest) '.']);

if (defs.plotOn == true)
	plotTraining(Costs(1:i), figNum, plotTitle);
end

end
