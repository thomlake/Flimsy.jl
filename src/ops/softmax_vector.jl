
# -- Softmax over vector of 1xN -- #
type ReverseVectorSoftmax{T<:GradVariable} <: ReverseOperation
    ys::Vector{T}
    xs::Vector{T}
end

function call(rop::ReverseVectorSoftmax)
    ys = rop.ys
    xs = rop.xs
    m = length(ys)
    n = length(ys[1].data)
    for k = 1:n
        for i = 1:m
            for j = 1:m
                if i == j
                    xs[i].grad[k] += ys[i].data[k] * (1 - ys[j].data[k]) * ys[j].grad[k]
                else
                    xs[i].grad[k] -= ys[i].data[k] * ys[j].data[k] * ys[j].grad[k]
                end
            end
        end
    end
    return nothing
end

function softmax{V<:Variable}(xs::Vector{V})
    m = length(xs)
    k, n = size(xs[1])
    k == 1 || error("softmax can only be applied over a vector of 1xN matrices")
    xmax = deepcopy(xs[1].data)
    for i = 2:m
        k_i, n_i = size(xs[i])
        k_i == 1 || error("softmax can only be applied over a vector of 1xN matrices")
        n_i == n || error("softmax can only be applied over a vector of matrices with the same size")
        for j = 1:n
            xmax[j] = max(xmax[j], xs[i].data[j])
        end
    end
    
    ys = [zero(x) for x in xs]
    Z = zero(xmax)
    for i = 1:m
        for j = 1:n
            ys[i].data[j] = exp(xs[i].data[j] - xmax[j])
            Z[j] += ys[i].data[j]
        end
    end
    for i = 1:m
        for j = 1:n
            ys[i].data[j] = ys[i].data[j] / Z[j] 
        end
    end
    return ys
end

function softmax{V<:GradVariable}(stack::CallbackStack, xs::Vector{V})
    m = length(xs)
    k, n = size(xs[1])
    k == 1 || error("softmax can only be applied over a vector of 1xN matrices")
    xmax = deepcopy(xs[1].data)
    for i = 2:m
        k_i, n_i = size(xs[i])
        k_i == 1 || error("softmax can only be applied over a vector of 1xN matrices")
        n_i == n || error("softmax can only be applied over a vector of matrices with the same size")
        for j = 1:n
            xmax[j] = max(xmax[j], xs[i].data[j])
        end
    end
    
    ys = [zero(x) for x in xs]
    Z = zero(xmax)
    for i = 1:m
        for j = 1:n
            ys[i].data[j] = exp(xs[i].data[j] - xmax[j])
            Z[j] += ys[i].data[j]
        end
    end
    for i = 1:m
        for j = 1:n
            ys[i].data[j] = ys[i].data[j] / Z[j] 
        end
    end
    push_callback!(stack, ReverseVectorSoftmax(ys, xs))
    return ys
end
