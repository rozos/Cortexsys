% Costs, i, PrintIter, figNum, plotTitle, trainError, testError
% (last two arguments are optional)
function plotTraining(varargin)
   %% Handle function arguments
    % Check number of arguments
    if ~(numel(varargin) == 3) && ~(numel(varargin) == 5)
        disp('ERROR, plotTraining(): incorrect number of arguments!');
        return;
    end

    if (numel(varargin) >= 3)
        Costs = varargin{1};
        figNum = varargin{2};
        plotTitle = varargin{3};
        plotError = false;
    end
    if (numel(varargin) == 5)
        trainError = varargin{4};
        testError = varargin{5};
        plotError = true;
    end    
   %%
    
   %set(0, 'CurrentFigure', figNum); % prevents figure from being brought to front
   figure(figNum);
   
   if ~plotError
       semilogy(Costs);
       grid minor
       title(plotTitle);
       ylabel('Cost');
       xlabel('Iteration');
       axis tight
   else
       subplot(2,1,1)
       semilogy(Costs);
       grid minor
       title(plotTitle);
       ylabel('Cost');
       axis tight
   
       subplot(2,1,2)
       x = linspace(1,numel(Costs), numel(trainError));
       plot(x,trainError, x, testError);
       grid minor
       ylabel('Error (%)');
       xlabel('Iteration');
       axis tight
       legend('Train','Test');
   end

   drawnow
end