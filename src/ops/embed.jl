
type ReverseEmbed{T<:GradVariable} <: ReverseOperation
    y::T
    w::T
    x::Vector{Vector{Int}}
end

function call{T}(rop::ReverseEmbed{T})
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

function embed!(y::AbstractArray, w::AbstractArray, x::Vector{Vector{Int}})
    for (k, x_k) in enumerate(x)
        for j in x_k
            for i = 1:size(w, 1)
                y[i,k] += w[i,j]
            end
        end
    end
    return y
end

embed!(y::AbstractArray, w::AbstractArray, x::Vector{Int}) = embed!(y, w, Vector{Int}[x])

embed(w::AbstractArray, x::Vector{Vector{Int}}) = embed!(zeros(size(w, 1), length(x)), w, x)

embed(w::AbstractArray, x::Vector{Int}) = embed(w, Vector{Int}[x])

function embed(scope::Scope, w::Variable, x::Vector{Vector{Int}})
    sz = size(w, 1), length(x)
    y_data = allocate(scope, eltype(w.data), sz, 0)
    return DataVariable(embed!(y_data, w.data, x))
end

embed(scope::Scope, w::Variable, x::Vector{Int}) = embed(scope, w, Vector{Int}[x])

function embed(scope::GradScope, w::GradVariable, x::Vector{Vector{Int}})
    sz = size(w, 1), length(x)
    y_data = allocate(scope, eltype(w.data), sz, 0)
    y = GradVariable(embed!(y_data, w.data, x), similar(scope, y_data, 0))
    push_callback!(scope, ReverseEmbed(y, w, x))
    return y
end

embed(scope::GradScope, w::GradVariable, x::Vector{Int}) = embed(scope, w, Vector{Int}[x])
