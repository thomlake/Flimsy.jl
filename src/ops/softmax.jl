
type ReverseSoftmax{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseSoftmax)
    y = rop.y
    x = rop.x
    for n = 1:size(y, 2)
        for i = 1:size(y, 1)
            for j = 1:size(y, 1)
                if i == j
                    x.grad[i,n] += y.data[i,n] * (1 - y.data[j,n]) * y.grad[j,n]
                else
                    x.grad[i,n] -= y.data[i,n] * y.data[j,n] * y.grad[j,n]
                end
            end
        end
    end
    return nothing
end

function softmax(x::AbstractArray)
    xmax = maximum(x, 1)
    y = zero(x)
    Z = zero(xmax)
    for j = 1:size(x, 2)
        for i = 1:size(x, 1)
            y[i,j] = exp(x[i,j] - xmax[j])
            Z[j] += y[i,j]
        end
    end
    for j = 1:size(x, 2)
        for i = 1:size(x, 1)
            y[i,j] = y[i,j] / Z[j] 
        end
    end
    return y
end

softmax(x::Variable) = DataVariable(softmax(x.data))

softmax(stack::CallbackStack, x::DataVariable) = DataVariable(softmax(x.data))

function softmax(stack::CallbackStack, x::GradVariable)
    y = GradVariable(softmax(x.data))
    push!(stack, ReverseSoftmax(y, x))
    return y
end
