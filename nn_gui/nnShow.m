function [] = nnShow(figNum, layers, defs)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    % Use Matlab's vector field graphic device to draw an arrow
    drawArrow = @(x,y) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0, 'k', 'LineWidth', 1.5, 'MaxHeadSize',2);
    
    N_l = numel(layers.sz);
    
    h = figure(figNum);
    movegui('west');
    axes(gca);
    % Clear contents of these axes / plot window
    cla reset
    
    hold on
    for k=1:N_l
        sz = 3*N_l*layers.sz{k}(1)/layers.sz{1}(1);
            pos = [-sz/2 3*(k-1) sz 1];
            % Draw rectangle representing a layer
            rectangle('Position',pos,'Curvature',[0 0], 'FaceColor', 'y', 'LineWidth', 1.5)
            
            if k<N_l
                %% Feed forward connections
                drawArrow([0 0], [3*(k-1)+1 3*(k)]);
                % Remove singleton dimensions from layer sizes
                sz_l_out = layers.sz{k+1}(layers.sz{k+1} ~= 1);
                sz_l_in = layers.sz{k}(layers.sz{k} ~= 1);
                % Draw dimensionality of parameter matrix
                text(0.5, 3*(k-1)+2, ['W_{ij} \in \bf{R}^{(' num2str(sz_l_out) ') x (' num2str(sz_l_in) ')}']);
                
                %% Recurrent connections
                if (layers.typ{k} == defs.TYPES.RECURRENT) || (layers.typ{k} == defs.TYPES.LSTM)
                    rectangle('Position',[-3 pos(2)-1 3 3],'Curvature',[1 1], 'LineWidth', 1.5, 'EdgeColor', 'b')
                    rectangle('Position',[-1.5 pos(2)+.05 3 pos(4)-.1],'Curvature',[0 0], 'FaceColor', 'y', 'EdgeColor', 'none')
                    text(-10, 3*(k-1)+1.5, ['W_{ij} \in \bf{R}^{' num2str(layers.sz{k}(1)) ' x ' num2str(layers.sz{k}(1)) '}']);
                end
            end
            
            % Draw activation function type within layer
            if layers.typ{k} == defs.TYPES.AVERAGE_POOLING
                layerName = 'Average Pooling';
            elseif layers.typ{k} == defs.TYPES.CONVOLUTIONAL
                layerName = [layers.af{k}.name ' (ConvNet)'];
            elseif k==1
                layerName = 'Input';
            else
                layerName = layers.af{k}.name;
            end
            text(0, 3*(k-1)+.5, layerName, 'HorizontalAlignment', 'center',...
                        'BackgroundColor', 'y', 'Margin', 0.1, 'FontSize', 9);
    end
    hold off
    axis square
    axis off
end

