
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

sigmoid(x::Variable) = DataVariable(sigmoid(x.data))

sigmoid(stack::CallbackStack, x::DataVariable) = DataVariable(sigmoid(x.data))

function sigmoid(stack::CallbackStack, x::GradVariable)
    y = GradVariable(sigmoid(x.data))
    push!(stack, ReverseSigmoid(y, x))
    return y
end
