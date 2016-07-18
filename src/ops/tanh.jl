
type ReverseTanh <: ReverseOperation
    y::Variable
    x::Variable
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

Base.tanh(scope::Scope, x::AbstractValue) = Constant(tanh!(similar(x.data), x.data))

function Base.tanh(scope::GradScope, x::Variable)
    y_data = similar(x.data)
    tanh!(y_data, x.data)
    y = Variable(y_data)
    push_callback!(scope, ReverseTanh(y, x))
    return y
end
