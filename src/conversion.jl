
function Base.convert(::Type{Vector}, params::Component)
    param_vector = GradVariable[]
    for f in fieldnames(params)
        T = fieldtype(typeof(params), f)
        if T <: GradVariable
            push!(param_vector, getfield(params, f))
        elseif T <: AbstractArray && eltype(T) <: GradVariable
            append!(param_vector, getfield(params, f)[:])
        elseif T <: Component
            append!(param_vector, convert(Vector, getfield(params, f)))
        elseif T <: AbstractArray && eltype(T) <: Component
            field = getfield(params, f)
            for component in field
                append!(param_vector, convert(Vector, component))
            end
        end
    end
    return param_vector
end

function Base.convert{C<:Component}(::Type{Dict}, params::C, prefix::ASCIIString="")
    if length(prefix) > 0
        prefix = string(prefix, ".")
    end
    param_dict = Dict{ASCIIString,Any}()
    for name in fieldnames(params)
        T = fieldtype(typeof(params), name)
        if T <: GradVariable
            key = string(prefix, name)
            param_dict[key] = getfield(params, name)
        elseif T <: AbstractArray && eltype(T) <: GradVariable
            key = string(prefix, name)
            param_dict[key] = getfield(params, name)
        elseif T <: Component
            key = string(prefix, name)
            param_dict[key] = convert(Dict, getfield(params, name), key)
        elseif T <: AbstractArray && eltype(T) <: Component
            key = string(prefix, name)
            field = getfield(params, name)
            param_dict[key] = [convert(Dict, field[i], key) for i = 1:endof(field)] 
        end
    end
    return param_dict
end
