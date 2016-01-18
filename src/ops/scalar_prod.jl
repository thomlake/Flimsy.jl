
type ReverseScalarProd{T<:GradVariable,F<:Real} <: ReverseOperation
    c::T
    a::F
    b::T
end

function call(rop::ReverseScalarProd)
    c = rop.c
    a = rop.a
    b = rop.b
    for i in eachindex(c)
        b.grad[i] += a * c.grad[i]
    end
    return nothing
end

Base.prod{V<:Variable}(a::Real, b::V) = V(a .* b.data)

function Base.prod{V<:Variable}(stack::CallbackStack, a::Real, b::V)
    y = prod(a, b)
    push_callback!(stack, ReverseScalarProd(y, a, b))
    return y
end

Base.prod{V<:Variable}(b::V, a::Real) = prod(a, b)

Base.prod(stack::CallbackStack, b::GradVariable, a::Real) = prod(stack, a, b)
