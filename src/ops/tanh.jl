
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

# Base.tanh(x::Variable) = DataVariable(tanh(x.data))

# Base.tanh(stack::CallbackStack, x::DataVariable) = tanh(x)

# function Base.tanh(stack::CallbackStack, x::GradVariable)
#     y = GradVariable(tanh(x.data))
#     push!(stack, ReverseTanh(y, x))
#     return y
# end

function tanh!(y::AbstractArray, x::AbstractArray)
    for i in eachindex(x)
        y[i] = tanh(x[i])
    end
end

Base.tanh(x::Variable) = DataVariable(tanh(x.data))

function Base.tanh(scope::Scope, x::DataVariable)
    y_data = similar(scope, x.data)
    tanh!(y_data, x.data)
    return DataVariable(y_data)
end

function Base.tanh(scope::Scope, x::GradVariable)
    y_data = similar(scope, x.data)
    y_grad = similar(scope, x.data)
    
    tanh!(y_data, x.data)
    y = GradVariable(y_data, y_grad)
    push!(scope.stack, ReverseTanh(y, x))
    return y
end
