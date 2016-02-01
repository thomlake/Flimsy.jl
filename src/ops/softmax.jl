
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

function softmax!(y::AbstractArray, x::AbstractArray)
    xmax = maximum(x, 1)
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

softmax(x::AbstractArray) = softmax!(similar(x), x)

softmax(scope::Scope, x::Variable) = DataVariable(softmax!(similar(scope, x.data), x.data))

function softmax(scope::GradScope, x::GradVariable)
    y_data = similar(scope, x.data)
    y_grad = similar(scope, y_data, 0)
    softmax!(y_data, x.data)
    y = GradVariable(y_data, y_grad)
    push_callback!(scope, ReverseSoftmax(y, x))
    return y
end
