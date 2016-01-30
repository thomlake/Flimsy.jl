
function call{T<:Component}(::Type{T}; kwargs...)
    @assert length(kwargs) == length(fieldnames(T))
    kwdict = Dict(kwargs)
    args = []
    for field in fieldnames(T)
        value = kwdict[field]
        ftype = fieldtype(T, field)
        if ftype <: Variable
            push!(args, typeof(value) <: Variable ? value : GradVariable(value))
        elseif ftype <: AbstractArray && eltype(ftype) <: Variable
            push!(args, eltype(value) <: Variable ? value : map(GradVariable, value))
        elseif ftype <: AbstractArray && length(ftype.parameters) > 0 && ftype.parameters[1].ub <: Variable
            push!(args, eltype(value) <: Variable ? value : map(GradVariable, value))
        else
            push!(args, value)
        end
    end
    return T{GradVariable}(args...)
end
