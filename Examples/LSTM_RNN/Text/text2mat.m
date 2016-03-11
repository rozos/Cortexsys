function X = text2mat(filepath, m, Nchars, varargin)
% TEXT2MAT: generate interleaved tensor representing numerical values of
% characters in a text file.
%
% text2mat(filepath, m, Nchars, varargin)
% varargin = {'bytes', 'binary', 'onehot'}
% Choose between decimal byte representation, 7-bit binary vector
% representaiton or 128-dimension "one hot" vector representation.
% Output format is [char value, line num, character]

    % char vec, training example number, sequence position (t)
    X = zeros(1,m,Nchars, 'uint8');

    fileID = fopen(filepath,'r');
    
    padVec = uint8(zeros(Nchars,1))'; % Zero is a null in ASCII
    
    % Read two line at a time, interleaving.
    % Eg, if we have lines 1,2,3,4; read: 12, 23, 34
    line1 = fgets(fileID);
    line2 = fgets(fileID);
    lineCnt = 2;
    for ex=1:m
        lineCnt = lineCnt+1;
        line3 = fgets(fileID);
        if ~ischar(line3)
            disp(['Reached end of file at line ' num2str(lineCnt)]);
            break;
        end

        bytes = uint8([line1 line2 line3]);
        bytesPad = [bytes padVec(numel(bytes)+1:end)];
        X(1,ex,:) = bytesPad;

        % Shift lines to one slot earlier
        line1 = line2;
        line2 = line3;
    end
    
    fclose(fileID);
    
    % Strip off unusued portion of matrix
    if (lineCnt < m)
        X = X(:,1:lineCnt,:);
    end
    
    % Convert to desired format
    if (numel(varargin) == 1)
        switch lower(varargin{1})
        case 'bytes'
            % do nothing, already in unit8 byte format
        case 'binary'
            X = uint8(dec2bin(squeeze(X)',7))';
            X(X==48) = 0;
            X(X==49) = 1;
            X = reshape(X,7,Nchars,[]);
            X = permute(X, [1 3 2]);
        case 'onehot'
            % Build up cell array where cell array contains time axis
            %%{
            Xca = cell(Nchars,1);
            OHeye = speye(128); 
            
            for t=1:Nchars
                Xt = squeeze(X(:,:,t))';
                Xca{t} = OHeye(1+Xt(:),:)';
            end
            X = Xca;
            %}
            
            %{
            Xtmp = eye(128);
            Xsq = squeeze(X)';
            Xtmp = Xtmp(1+Xsq(:),:);
            X = reshape(Xtmp',128,Nchars,[]);
            X = permute(X, [1 3 2]);
            %}
        otherwise
            error('Unknown format. Valid options are bytes, binary or onehot.')
        end
    end
end