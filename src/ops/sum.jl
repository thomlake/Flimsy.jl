
type ReverseSum{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseSum)
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] += y.grad[i]
    end
    return nothing
end

type ReverseColumnSum{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseColumnSum)
    y = rop.y
    x = rop.x
    for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            x.grad[i] += y.grad[i,j]
        end
    end
    return nothing
end

Base.sum{V<:Variable}(a::V, b::V) = V(a.data .+ b.data)

function Base.sum(stack::CallbackStack, a::GradVariable, b::GradVariable)
    c = sum(a, b)
    if size(a) == size(b)
        push_callback!(stack, ReverseSum(c, a))
        push_callback!(stack, ReverseSum(c, b))
    elseif is_matrix(a) && is_column_vector(b)
        push_callback!(stack, ReverseSum(c, a))
        push_callback!(stack, ReverseColumnSum(c, b))
    elseif is_matrix(b) && is_column_vector(a)
        push_callback!(stack, ReverseSum(c, b))
        push_callback!(stack, ReverseColumnSum(c, a))
    else
        error("no sum for sizes a: $(size(a)), b: $(size(b))")
    end
    return c
end

# # -- Sum (arbitrary number of blocks) -- #
function Base.sum{V<:Variable}(xs::Vector{V})
    y = sum(xs[1], xs[2])
    for i = 3:length(xs)
        y = sum(y, xs[i])
    end
    return y
end

function Base.sum{V<:Variable}(stack::CallbackStack, xs::Vector{V})
    y = sum(stack, xs[1], xs[2])
    for i = 3:length(xs)
        y = sum(stack, y, xs[i])
    end
    return y
end

Base.sum{V<:Variable}(x1::V, x2::V, xrest::V...) = sum([x1, x2, xrest...])

Base.sum(stack::CallbackStack, x1::GradVariable, x2::GradVariable, xrest::GradVariable...) = sum(stack, [x1, x2, xrest...])

