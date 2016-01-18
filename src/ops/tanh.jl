
type ReverseTanh{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseTanh)
    y = rop.y
    x = rop.x
    for i in eachindex(x)
        x.grad[i] += (1 - (y.data[i] * y.data[i])) * y.grad[i]
    end
    return nothing
end

Base.tanh{V<:Variable}(x::V) = V(tanh(x.data))

function Base.tanh(stack::CallbackStack, x::GradVariable)
    y = tanh(x)
    push_callback!(stack, ReverseTanh(y, x))
    return y
end
