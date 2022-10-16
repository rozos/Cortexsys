%% Generate some text by 'sampling' from LSTM
T_samp_temp = 1/35; % "Temperature" of random sampling process (higher temperatures lead to more randomness)
T_samp = 5000; % Length of sequence to generate
seedText = 'ROMEO:';

% Initial text (to provide some context)
vsize = numel(vmap);
Xs = full(ascii2onehot(seedText, vmap));
Xs = [zeros(vsize,1) Xs];
Xtmp = zeros(vsize,1,numel(seedText)+1);
Xtmp(:,1,:) = Xs;
Xs = Xtmp;

nn.disableCuda();
nn.A{1} = varObj(Xs, nn.defs);
preallocateMemory(nn, 1, numel(seedText)+2);
% Load up the LSTM with the context
for t=2:numel(seedText)+1
    feedforwardLSTM(nn, 1, t, false, true);
    [~,cout] = max(nn.A{end}.v(:,1,t));
    fprintf('%s', vmap(cout));
end

% Start sampling characters by feeding output back into the input for the
% next time step
for t=numel(seedText)+2:numel(seedText)+T_samp
    % Generate a random sample from the softmax probability distribution
    % First, adjust/scale the distribution by a "temperature" that controls
    % how likely we are to pick the maximum likelihood prediction
    P_next_char = exp(1/T_samp_temp*nn.A{end}.v(:,1,t-1));
    P_next_char = P_next_char./sum(P_next_char); % normalize distribution
    cin = randsample(vsize,1,true,P_next_char);
    fprintf('%s', vmap(cin));
    
    % Plot the distribution over characters
    %{
    figure(777);
    plot(P_next_char);
    set(gca, 'XTick',1:numel(P_next_char), 'XTickLabel',vmap)
    waitforbuttonpress;
    %}
    
    % Feed back the output to the input
    % Generate the input for the next time step
    tmp = zeros(vsize,1);
    tmp(cin) = 1;
    nn.A{1}.v(:,1,t) = tmp;
    
    % Step the RNN forward
    feedforwardLSTM(nn, 1, t, false, true);
    
end
disp('');
