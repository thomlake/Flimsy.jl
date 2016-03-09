
function Base.convert{V<:Variable}(::Type{Vector}, params::Component{V})
    param_vector = V[]
    for f in fieldnames(params)
        T = fieldtype(typeof(params), f)
        if T <: Variable
            push!(param_vector, getfield(params, f))
        elseif T <: AbstractArray && eltype(T) <: Variable
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

Base.convert(::Type{Vector}, model::Model) = convert(Vector, model.component)

function Base.convert{C<:Component}(::Type{Dict}, params::C, prefix::ASCIIString=string(C.name))
    dict = Dict{ASCIIString,Any}()
    for name in fieldnames(params)
        T = fieldtype(typeof(params), name)
        if T <: Variable
            key = join([prefix, name], ".")
            dict[key] = getfield(params, name)
        elseif T <: AbstractArray && eltype(T) <: Variable
            key = join([prefix, name], ".")
            dict[key] = getfield(params, name)
        elseif T <: Component
            key = join([prefix, name], ".")
            dict[key] = convert(Dict, getfield(params, name), key)
        elseif T <: AbstractArray && eltype(T) <: Component
            key = join([prefix, name], ".")
            field = getfield(params, name)
            dict[key] = [convert(Dict, field[i], key) for i = 1:endof(field)] 
        end
    end
    return dict
end

Base.convert(::Type{Dict}, model::Model) = convert(Dict, model.component)
