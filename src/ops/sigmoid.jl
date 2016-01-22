
type ReverseSigmoid{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseSigmoid{T}) = nothing

function call{T<:GradVariable}(rop::ReverseSigmoid{T})
    y = rop.y
    x = rop.x
    for i in eachindex(x)
        x.grad[i] += y.data[i] * (1 - y.data[i]) * y.grad[i]
    end
    return nothing
end

function sigmoid(x::AbstractArray)
    y = zero(x)
    for i in eachindex(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    return y
end

sigmoid(x::Variable) = DataVariable(sigmoid(x.data))

@generated function sigmoid{T<:Variable}(stack::CallbackStack, x::T)
    if T <: GradVariable
        return quote
            y = GradVariable(sigmoid(x.data))
            push_callback!(stack, ReverseSigmoid(y, x))
            return y
        end
    else
        return :(DataVariable(sigmoid(x.data)))
    end
end
