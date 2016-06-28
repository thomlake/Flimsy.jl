
is_variable(T) = T <: Variable

function is_variable_array(T)
    if T <: AbstractArray && eltype(T) <: Variable
        return true
    end
    try 
        return T <: AbstractArray && length(T.parameters) > 0 && T.parameters[1].ub <: Variable 
    catch
        return false
    end
end


function call{T<:Component}(::Type{T}; kwargs...)
    @assert length(kwargs) == length(fieldnames(T))
    kwdict = Dict(kwargs)
    args = []
    for field in fieldnames(T)
        value = kwdict[field]
        ftype = fieldtype(T, field)
        if is_variable(ftype)
            push!(args, typeof(value) <: Variable ? value : GradVariable(value, zero(value)))
        elseif is_variable_array(ftype)
            push!(args, eltype(value) <: Variable ? value : map(v -> GradVariable(v, zero(v)), value))
        else
            push!(args, value)
        end
    end
    return T(args...)
end