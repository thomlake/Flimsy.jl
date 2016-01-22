
function call{T<:Component}(::Type{T}; kwargs...)
    @assert length(kwargs) == length(fieldnames(T))
    kwdict = Dict(kwargs)
    args = []
    for field in fieldnames(T)
        value = kwdict[field]
        if fieldtype(T, field) <: Variable
            push!(args, typeof(value) <: Variable ? value : DataVariable(value))
        else
            push!(args, value)
        end
    end
    return T(args...)
end

@generated function GradComponent{C<:Component}(theta::C)
    CType = C.name.primary
    return quote
        args = Any[]
        for f in fieldnames(theta)
            T = fieldtype(C, f)
            if T <: Variable
                push!(args, GradVariable(getfield(theta, f).data))
            elseif T <: AbstractArray && eltype(T) <: Variable
                push!(args, map(x -> GradVariable(x.data), getfield(theta, f)))
            elseif T <: Component
                push!(args, GradComponent(getfield(theta, f)))
            elseif T <: AbstractArray && eltype(T) <: Component
                push!(args, map(GradComponent, getfield(theta, f)))
            end
        end
        return $CType(args...)
    end
end

function getparams(theta::Component)
    params = Variable[]
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

function getnamedparams(theta::Component)
    params = Tuple{Any,Variable}[]
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
