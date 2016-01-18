type ReverseBroadcastProd{T<:GradVariable} <: ReverseOperation
    c::T
    a::T
    b::T
end

function call(rop::ReverseBroadcastProd)
    c = rop.c
    a = rop.a
    b = rop.b
    for j = 1:size(c, 2)
        for i = 1:size(c, 1)
            a.grad[j] += c.grad[i,j] * b.data[i,j]
            b.grad[i,j] += c.grad[i,j] * a.data[j]
        end
    end
    # for j = 1:size(x, 2)
    #     for i = 1:size(x, 1)
    #         x.grad[i,j] += y.grad[i,j] .* a.data[j]
    #     end
    # end
    return nothing
end

type ReverseProd{T<:GradVariable} <: ReverseOperation
    c::T
    a::T
    b::T
end

function call(rop::ReverseProd)
    c = rop.c
    a = rop.a
    b = rop.b
    for j = 1:size(c, 2)
        for i = 1:size(c, 1)
            a.grad[i,j] += c.grad[i,j] * b.data[i,j]
            b.grad[i,j] += c.grad[i,j] * a.data[i,j]
        end
    end
    return nothing
end

Base.prod{V<:Variable}(a::V, b::V) = V(a.data .* b.data)

function Base.prod(stack::CallbackStack, a::GradVariable, b::GradVariable)
    c = prod(a, b)
    if size(a) == size(b)
        push_callback!(stack, ReverseProd(c, a, b))
    elseif is_matrix(a) && is_row_vector(b)
        push_callback!(stack, ReverseBroadcastProd(c, b, a))
    elseif is_row_vector(a) && is_matrix(b)
        push_callback!(stack, ReverseBroadcastProd(c, a, b))
    elseif is_column_vector(a) && is_scalar(b)
        push_callback!(stack, ReverseBroadcastProd(c, b, a))
    elseif is_scalar(a) && is_column_vector(b)
        push_callback!(stack, ReverseBroadcastProd(c, a, b))
    else
        error("no prod for sizes a: $(size(a)), b: $(size(b))")
    end
    return c
end
