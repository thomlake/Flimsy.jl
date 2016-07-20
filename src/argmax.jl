
argmax(x::Vector) = [indmax(x)]

function argmax(x::Matrix)
    n_rows, n_cols = size(x)
    imax = zeros(Int, n_cols)
    for j = 1:n_cols
        m = -Inf
        for i = 1:n_rows
            if x[i,j] > m
                m = x[i,j]
                imax[j] = i
            end
        end
    end
    return imax
end

argmax(x::AbstractValue) = argmax(x.data)


"""
Return the index of the maximum element not equal to k.

If x is a Vector and k is an Integer
    argmaxneq(x, k) = argmax { x_i : i != k }

If x is a Matrix and k is a Vector{Integer} operates column-wise
    argmaxneq(x, k) = [argmaxneq(x[:,1], k[1]), ..., argmaxneq(x[:,end], k[end])]
"""
function argmaxneq end

function argmaxneq(x::Vector, k::Integer)
    m = -Inf
    imax = 0
    for i in eachindex(x)
        if i != k && x[i] > m
            m = x[i]
            imax = i
        end
    end
    return imax
end

function argmaxneq{I<:Integer}(x::Matrix, ks::Vector{I})
    n_rows, n_cols = size(x)
    @assert n_cols == length(ks)
    imax = zeros(Int, n_cols)
    for j = 1:n_cols
        m = -Inf
        for i = 1:n_rows
            if i != ks[j] && x[i,j] > m
                m = x[i,j]
                imax[j] = i
            end
        end
    end
    return imax
end

argmaxneq{I<:Integer}(x::AbstractValue, ks::Vector{I}) = argmaxneq(x.data, ks)
