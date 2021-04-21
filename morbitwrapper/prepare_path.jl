
function prepare_path( s )
    if isempty(s)
        return "."
    else
        if isabspath( s ) 
            return s 
        else 
            if startswith( s , "@")
                return Base.load_path_expand(s)
            else
                return abspath( joinpath( @__DIR__, "..", s ) )
            end
        end 
    end 
end
