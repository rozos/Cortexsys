function [X, vocabMap] = streamText2mat(filepath, Nchars, offset)
% TEXT2MAT: generate cell array of
%           offset specifies the shift in the time sequence. offset=1
%           corresponds to shifting one character later in time, thus
%           giving a target/prediction training set.
    fileID = fopen(filepath,'r');
    text = fread(fileID, '*char');
    text = [text(1+offset:end); char(10*ones(offset,1))];
    
    % Determine how many training examples we get out of this text
    m = ceil(numel(text)/Nchars);
    % Pad text with spaces so it is multiple of number of characters per
    % training example
    text = [text; char(10*ones(m*Nchars-numel(text),1))];
    
    % Reshape into rows of length Nchars (corresponds to time axis)
    text = reshape(text,Nchars,[]);
    fclose(fileID);
  
    % Compute the vocabulary (identify unique characters)
    vocabMap = unique(text);
    vocabSize = numel(vocabMap);
    % Map all ASCII values to the reduced set
    for i=1:numel(vocabMap)
        text(text==vocabMap(i)) = i;
    end
    
    % Build up cell array where cell array contains time axis
    % This format is necessary because 3D sparse matricies are not
    % supported in Matlab
    Xca = cell(Nchars,1);
    I = speye(vocabSize); 

    for t=1:Nchars
        Xt = text(t,:);
        Xca{t} = I(Xt(:),:)';
    end
    X = Xca;
end