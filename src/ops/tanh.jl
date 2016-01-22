
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

Base.tanh(x::Variable)  = DataVariable(tanh(x.data))

Base.tanh(stack::CallbackStack, x::DataVariable) = tanh(x)

function Base.tanh(stack::CallbackStack, x::GradVariable)
    y = GradVariable(tanh(x.data))
    push!(stack, ReverseTanh(y, x))
    return y
end
