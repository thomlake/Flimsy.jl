
type ReverseMinus{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseMinus)
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] -= y.grad[i]
    end
    return nothing
end

type ReverseColumnMinus{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseColumnMinus)
    y = rop.y
    x = rop.x
    for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            x.grad[i] -= y.grad[i,j]
        end
    end
    return nothing
end

minus{V<:Variable}(a::V, b::V) = V(a.data .- b.data)

function minus(stack::CallbackStack, a::GradVariable, b::GradVariable)
    c = minus(a, b)
    if size(a) == size(b)
        push_callback!(stack, ReverseSum(c, a))
        push_callback!(stack, ReverseMinus(c, b))
    elseif is_matrix(a) && is_column_vector(b)
        push_callback!(stack, ReverseSum(c, a))
        push_callback!(stack, ReverseColumnMinus(c, b))
    elseif is_matrix(b) && is_column_vector(a)
        push_callback!(stack, ReverseMinus(c, b))
        push_callback!(stack, ReverseColumnSum(c, a))
    else
        error("no minus for sizes a: $(size(a)), b: $(size(b))")
    end
    return c
end
