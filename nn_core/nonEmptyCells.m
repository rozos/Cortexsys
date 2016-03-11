function N = nonEmptyCells(C)
%numelNonEmpty Simple function that returns the number of non-empty cells
%in a cell array.

    N = find(~cellfun(@isempty, C));
end

