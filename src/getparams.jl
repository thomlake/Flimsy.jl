
function getparams{V<:Variable}(theta::Component{V})
    params = V[]
    for f in fieldnames(theta)
        T = fieldtype(typeof(theta), f)
        if T <: Variable
            push!(params, getfield(theta, f))
        elseif T <: AbstractArray && eltype(T) <: Variable
            append!(params, getfield(theta, f)[:])
        elseif T <: Component
            append!(params, getparams(getfield(theta, f)))
        elseif T <: AbstractArray && eltype(T) <: Component
            field = getfield(theta, f)
            for component in field
                append!(params, getparams(component))
            end
        end
    end
    return params
end

Base.convert{V<:Variable}(::Type{Vector}, params::Component{V}) = getparams(params)

function getnamedparams{V<:Variable}(theta::Component{V})
    params = Tuple{Any,V}[]
    for name in fieldnames(theta)
        T = fieldtype(typeof(theta), name)
        if T <: Variable
            push!(params, (name, getfield(theta, name)))
        elseif T <: AbstractArray && eltype(T) <: Variable
            field = getfield(theta, name)
            for i = 1:endof(field)
                push!(params, ((name, i), field[i]))
            end
        elseif T <: Component
            for (subname, param) in getnamedparams(getfield(theta, name))
                push!(params, ((name, subname), param))
            end
        elseif T <: AbstractArray && eltype(T) <: Component
            field = getfield(theta, name)
            for i = 1:endof(field)
                for (subname, param) in getnamedparams(field[i])
                    push!(params, (((name, i), subname), param))
                end
            end
        end
    end
    return params
end

function getdictparams{C<:Component}(params::C, prefix::ASCIIString=string(C.name))
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
            dict[key] = getdictparams(getfield(params, name), key)
        elseif T <: AbstractArray && eltype(T) <: Component
            key = join([prefix, name], ".")
            field = getfield(params, name)
            dict[key] = [getdictparams(field[i], key) for i = 1:endof(field)] 
        end
    end
    return (C, dict)
end

Base.convert(::Type{Dict}, params::Component) = getdictparams(params)
