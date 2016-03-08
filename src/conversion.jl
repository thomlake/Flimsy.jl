
function Base.convert{V<:Variable}(::Type{Vector}, params::Component{V})
    params = V[]
    for f in fieldnames(theta)
        T = fieldtype(typeof(theta), f)
        if T <: Variable
            push!(params, getfield(theta, f))
        elseif T <: AbstractArray && eltype(T) <: Variable
            append!(params, getfield(theta, f)[:])
        elseif T <: Component
            append!(params, convert(Vector, getfield(theta, f)))
        elseif T <: AbstractArray && eltype(T) <: Component
            field = getfield(theta, f)
            for component in field
                append!(params, convert(Vector, component))
            end
        end
    end
    return params
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
