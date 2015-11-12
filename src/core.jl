typealias BPStack Vector{Function}

abstract Component

function backprop!(stack::BPStack)
    for i = endof(stack):-1:1
        stack[i]()
    end
    empty!(stack)
end

function gradient!(f::Function, theta::Component, args...)
    stack = BPStack()
    output = f(stack, theta, args...)
    backprop!(stack)
    return output
end

function getparams(theta::Component)
    params = Variable[]
    for f in fieldnames(theta)
        T = fieldtype(typeof(theta), f)
        if T <: AbstractVariable
            push!(params, getfield(theta, f))
        elseif T <: AbstractArray && eltype(T) <: AbstractVariable
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
        if T <: AbstractVariable
            push!(params, (name, getfield(theta, name)))
        elseif T <: AbstractArray && eltype(T) <: AbstractVariable
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
