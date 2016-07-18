
type ReverseConcat <: ReverseOperation
    y::Variable
    xs::Vector{AbstractValue}
end

function call(rop::ReverseConcat)
    y = rop.y
    xs = rop.xs
    offset = 0
    @flimsy_inbounds for k = 1:length(xs)
        if isa(xs[k], Variable)
            for j = 1:size(xs[k], 2)
                for i = 1:size(xs[k], 1)
                    xs[k].grad[i,j] += y.grad[offset + i,j]
                end
            end
        end
        offset += size(xs[k], 1)
    end
    return nothing
end

function concat!{T<:Matrix}(y::T, xs::Vector{T})
    offset = 0
    @flimsy_inbounds for k = 1:length(xs)
        for j = 1:size(xs[k], 2)
            for i = 1:size(xs[k], 1)
                y[offset + i,j] = xs[k][i,j]
            end
        end
        offset += size(xs[k], 1)
    end
    return y
end

concat{T<:Matrix}(xs::Vector{T}) = vcat(xs...)

function concat{T<:AbstractValue}(scope::Scope, xs::Vector{T})
    m, n = size(xs[1])
    for i = 2:length(xs)
        m_i, n_i = size(xs[i])
        m += m_i
        n == n_i || throw(OperationError("can only concatenate vectors with members having the same number of columns"))
    end
    xs_data = Matrix{FloatX}[x.data for x in xs]
    ys_data = Matrix{FloatX}(m, n)

    return Constant(concat!(ys_data, xs_data))
end

function concat{T<:AbstractValue}(scope::GradScope, xs::Vector{T})
    m, n = size(xs[1])
    for i = 2:length(xs)
        m_i, n_i = size(xs[i])
        m += m_i
        n == n_i || throw(OperationError("can only concatenate vectors with members having the same number of columns"))
    end
    xs_data = Matrix{FloatX}[x.data for x in xs]
    ys_data = Matrix{FloatX}(m, n)

    y = Variable(concat!(ys_data, xs_data))
    push_callback!(scope, ReverseConcat(y, xs))
    return y
end

function concat(scope::GradScope, xs::Vector{Constant})
    m, n = size(xs[1])
    for i = 2:length(xs)
        m_i, n_i = size(xs[i])
        m += m_i
        n == n_i || throw(OperationError("can only concatenate vectors with members having the same number of columns"))
    end
    xs_data = Matrix{FloatX}[x.data for x in xs]
    ys_data = Matrix{FloatX}(m, n)

    return Constant(concat!(ys_data, xs_data))
end

concat{V<:AbstractValue}(scope, xs::V...) = concat(scope, [x for x in xs])
