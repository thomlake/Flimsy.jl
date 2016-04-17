
type ReverseLinearEmbed <: ReverseOperation
    y::GradVariable
    w::GradVariable
    x::Vector{Vector{Int}}
end

function call(rop::ReverseLinearEmbed)
    y = rop.y
    w = rop.w
    x = rop.x

    for k = 1:length(x)
        for j in x[k]
            for i = 1:size(y, 1)
                w.grad[i,j] += y.grad[i,k]
            end
        end
    end

    return nothing
end

function linear!(y::AbstractArray, w::AbstractArray, x::Vector{Vector{Int}})
    for k = 1:length(x)
        for j in x[k]
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
    y_data = zeros(FloatX, size(w, 1), length(x))
    return DataVariable(linear!(y_data, w.data, x))
end

function linear(scope::GradScope, w::GradVariable, x::Vector{Vector{Int}})
    y_data = zeros(FloatX, size(w, 1), length(x))
    y = GradVariable(linear!(y_data, w.data, x), zero(y_data))
    push_callback!(scope, ReverseLinearEmbed(y, w, x))
    return y
end

linear(scope::Scope, w::Variable, x::Vector{Int}) = linear(scope, w, Vector{Int}[x])

linear(scope::Scope, w::Variable, x::Int) = linear(scope, w, Int[x])
