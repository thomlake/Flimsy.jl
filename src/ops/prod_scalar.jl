
type ReverseScalarProd{T<:Variable,F<:Real} <: ReverseOperation
    c::T
    a::F
    b::T
end

call{T<:DataVariable,F}(rop::ReverseScalarProd{T,F}) = nothing

function call{T<:GradVariable,F}(rop::ReverseScalarProd{T,F})
    c = rop.c
    a = rop.a
    b = rop.b
    for i in eachindex(c)
        b.grad[i] += a * c.grad[i]
    end
    return nothing
end

Base.prod(a::Real, b::Variable) = DataVariable(a .* b.data)

function Base.prod(stack::CallbackStack, a::Real, b::GradVariable)
    y = GradVariable(a .* b.data)
    push_callback!(stack, ReverseScalarProd(y, a, b))
    return y
end

Base.prod(stack::CallbackStack, a::Real, b::DataVariable) = DataVariable(a .* b.data)

Base.prod(b::Variable, a::Real) = prod(a, b)

Base.prod(stack::CallbackStack, b::Variable, a::Real) = prod(stack, a, b)
