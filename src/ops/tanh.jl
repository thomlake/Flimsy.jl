
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

function tanh!(y::AbstractArray, x::AbstractArray)
    for i in eachindex(x)
        y[i] = tanh(x[i])
    end
    return y
end

Base.tanh(scope::Scope, x::Variable) = DataVariable(tanh!(similar(scope, x.data), x.data))

function Base.tanh(scope::GradScope, x::GradVariable)
    y_data = similar(scope, x.data)
    y_grad = similar(scope, y_data, 0)
    tanh!(y_data, x.data)
    y = GradVariable(y_data, y_grad)
    push_callback!(scope, ReverseTanh(y, x))
    return y
end
