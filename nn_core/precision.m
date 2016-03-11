function x = precision(x, defs)
    
    if isa(x, 'varObj')
        if strcmp('single', defs.PRECISION)
            x.v = single(x.v);
        elseif strcmp('double', defs.PRECISION)
            x.v = double(x.v);
        else
            disp('ERROR! precision(): incorrect precision specified!');
        end
    else
        if strcmp('single', defs.PRECISION)
            x = single(x);
        elseif strcmp('double', defs.PRECISION)
            x = double(x);
        else
            disp('ERROR! precision(): incorrect precision specified!');
        end
    end
end