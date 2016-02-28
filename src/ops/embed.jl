
type ReverseLinearEmbed{T<:GradVariable} <: ReverseOperation
    y::T
    w::T
    x::Vector{Vector{Int}}
end

function call{T}(rop::ReverseLinearEmbed{T})
    y = rop.y
    w = rop.w
    x = rop.x

    for (k, x_k) in enumerate(x)
        for j in x_k
            for i = 1:size(y, 1)
                w.grad[i,j] += y.grad[i,k]
            end
        end
    end

    return nothing
end

function linear!(y::AbstractArray, w::AbstractArray, x::Vector{Vector{Int}})
    for (k, x_k) in enumerate(x)
        for j in x_k
            for i = 1:size(w, 1)
                y[i,k] += w[i,j]
            end
        end
    end
    return y
end

linear!(y::AbstractArray, w::AbstractArray, x::Vector{Int}) = linear!(y, w, Vector{Int}[x])

linear(w::AbstractArray, x::Vector{Vector{Int}}) = linear!(zeros(size(w, 1), length(x)), w, x)

linear(w::AbstractArray, x::Vector{Int}) = linear(w, Vector{Int}[x])

linear(w::AbstractArray, x::Int) = linear(w, Int[x])

function linear(scope::Scope, w::Variable, x::Vector{Vector{Int}})
    sz = size(w, 1), length(x)
    y_data = allocate(scope, eltype(w.data), sz, 0)
    return DataVariable(linear!(y_data, w.data, x))
end

function linear(scope::GradScope, w::GradVariable, x::Vector{Vector{Int}})
    sz = size(w, 1), length(x)
    y_data = allocate(scope, eltype(w.data), sz, 0)
    y = GradVariable(linear!(y_data, w.data, x), similar(scope, y_data, 0))
    push_callback!(scope, ReverseLinearEmbed(y, w, x))
    return y
end

linear(scope::Scope, w::Variable, x::Vector{Int}) = linear(scope, w, Vector{Int}[x])

linear(scope::Scope, w::Variable, x::Int) = linear(scope, w, Int[x])
