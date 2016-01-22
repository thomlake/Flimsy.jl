
type ReverseSoftmax{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseSoftmax{T}) = nothing

function call{T<:GradVariable}(rop::ReverseSoftmax{T})
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

@generated function softmax{T<:Variable}(stack::CallbackStack, x::T)
    if T <: GradVariable
        return quote
            y = GradVariable(softmax(x.data))
            push_callback!(stack, ReverseSoftmax(y, x))
            return y
        end
    else
        return :(DataVariable(softmax(x.data)))
    end
end
