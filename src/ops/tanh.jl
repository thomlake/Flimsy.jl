
type ReverseTanh <: ReverseOperation
    y::GradVariable
    x::GradVariable
end

function call(rop::ReverseTanh)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for i in eachindex(x)
        x.grad[i] += (1 - (y.data[i] * y.data[i])) * y.grad[i]
    end
    return nothing
end

function tanh!(y::AbstractArray, x::AbstractArray)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = tanh(x[i])
    end
    return y
end

Base.tanh(scope::Scope, x::Variable) = DataVariable(tanh!(similar(x.data), x.data))

function Base.tanh(scope::GradScope, x::GradVariable)
    y_data = similar(x.data)
    y_grad = zero(y_data)
    tanh!(y_data, x.data)
    y = GradVariable(y_data, y_grad)
    push_callback!(scope, ReverseTanh(y, x))
    return y
end
