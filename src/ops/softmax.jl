
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

softmax{V<:Variable}(x::V) = V(softmax(x.data))

function softmax(stack::CallbackStack, x::GradVariable)
    y = softmax(x)
    push_callback!(stack, ReverseSoftmax(y, x))
    return y
end

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
    ys = softmax(xs)
    push_callback!(stack, ReverseVectorSoftmax(ys, xs))
    return ys
end

# function softmax{V<:GradVariable}(xs::Vector{V})
#     all(x -> size(x) == (1, 1), xs) || error("softmax can only be applied over vectors if elements are of size (1, 1)")
#     xmax = -Inf
#     for x in xs
#         if x.data[1] > xmax
#             xmax = x.data[1]
#         end
#     end

#     Z = 0.0
#     ys = Variable{typeof(xs[1].data),1,1}[zero(x) for x in xs]
#     for i in eachindex(xs)
#         e_xi = exp(xs[i].data[1] - xmax)
#         ys[i].data[1] = e_xi
#         Z += e_xi
#     end

#     for i in eachindex(ys)
#         ys[i].data[1] /= Z
#     end

#     return ys
# end

# function softmax{V<:Variable}(stack::BPStack, xs::Vector{V})
#     ys = softmax(xs)
#     push!(stack, () -> bwd_softmax(ys, xs))
#     return ys
# end
