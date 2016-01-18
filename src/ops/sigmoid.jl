
type ReverseSigmoid{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseSigmoid)
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

sigmoid{V<:Variable}(x::V) = V(sigmoid(x.data))

function sigmoid(stack::CallbackStack, x::GradVariable)
    y = sigmoid(x)
    push_callback!(stack, ReverseSigmoid(y, x))
    return y
end
