
type ReverseSoftmax <: ReverseOperation
    y::GradVariable
    x::GradVariable
end

function call(rop::ReverseSoftmax)
    y = rop.y
    x = rop.x
    x_grad = x.grad
    y_data = y.data
    y_grad = y.grad
    n_rows, n_cols = size(y)

    @flimsy_inbounds for n = 1:n_cols
        for i = 1:n_rows
            for j = 1:n_rows
                if i == j
                    x_grad[i,n] += y_data[i,n] * (1 - y_data[j,n]) * y_grad[j,n]
                else
                    x_grad[i,n] -= y_data[i,n] * y_data[j,n] * y_grad[j,n]
                end
            end
        end
    end
    
    return nothing
end

function softmax!{T<:AbstractFloat}(y::Matrix{T}, x::Matrix{T})
    m = size(x, 1)
    Z = zero(T)
    xmax = typemin(T)

    @flimsy_inbounds for j = 1:size(x, 2)
        Z = zero(T)
        xmax = typemin(T)

        for i = 1:m
            if x[i,j] > xmax
                xmax = x[i,j]
            end
        end
        
        for i = 1:m
            y[i,j] = exp(x[i,j] - xmax)
            Z += y[i,j]
        end
        
        for i = 1:m
            y[i,j] = y[i,j] / Z
        end
    end

    return y
end

softmax(x::Matrix) = softmax!(similar(x), x)

softmax(scope::Scope, x::Variable) = DataVariable(softmax!(similar(x.data), x.data))

function softmax(scope::GradScope, x::GradVariable)
    y_data = similar(x.data)
    y_grad = zero(y_data)
    softmax!(y_data, x.data)
    y = GradVariable(y_data, y_grad)
    push_callback!(scope, ReverseSoftmax(y, x))
    return y
end
