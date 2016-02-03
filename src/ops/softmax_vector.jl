
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


function softmax_vector!{T}(ys::Vector{Matrix{T}}, xs::Vector{Matrix{T}})
    m = length(xs)
    k, n = size(xs[1])
    k == 1 || throw(OperationError("softmax can only be applied over a vector of 1xN matrices"))
    xmax = deepcopy(xs[1])
    for i = 2:m
        k_i, n_i = size(xs[i])
        k_i == 1 || throw(OperationError("softmax can only be applied over a vector of 1xN matrices"))
        n_i == n || throw(OperationError("softmax can only be applied over a vector of matrices with the same size"))
        for j = 1:n
            xmax[j] = max(xmax[j], xs[i][j])
        end
    end
    
    Z = zeros(n)
    for i = 1:m
        for j = 1:n
            ys[i][j] = exp(xs[i][j] - xmax[j])
            Z[j] += ys[i][j]
        end
    end
    for i = 1:m
        for j = 1:n
            ys[i][j] = ys[i][j] / Z[j] 
        end
    end
    return ys
end

softmax{T<:AbstractMatrix}(xs::Vector{T}) = softmax!([zero(x) for x in xs], xs)

function softmax{T<:Variable}(scope::Scope, xs::Vector{T})
    E = eltype(T)
    xs_data = Matrix{E}[x.data for x in xs]
    ys_data = Matrix{E}[similar(scope, x_data) for x_data in xs_data]
    softmax_vector!(ys_data, xs_data)
    return DataVariable{E}[DataVariable(y_data) for y_data in ys_data]
end

function softmax{T<:GradVariable}(scope::GradScope, xs::Vector{T})
    E = eltype(T)
    xs_data = Matrix{E}[x.data for x in xs]
    ys_data = Matrix{E}[similar(scope, x_data) for x_data in xs_data]
    softmax_vector!(ys_data, xs_data)
    ys = GradVariable{E}[GradVariable(y_data, similar(scope, y_data, 0)) for y_data in ys_data]
    push_callback!(scope, ReverseVectorSoftmax(ys, xs))
    return ys
end
