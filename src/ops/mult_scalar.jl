
type ReverseScalarMult{T<:GradVariable,F<:Real} <: ReverseOperation
    c::T
    a::F
    b::T
end

function call(rop::ReverseScalarMult)
    c = rop.c
    a = rop.a
    b = rop.b
    for i in eachindex(c)
        b.grad[i] += a * c.grad[i]
    end
    return nothing
end

mult(a::Real, b::AbstractArray) = a .* b

mult(a::Real, b::Variable) = DataVariable(mult(a, b.data))

mult(stack::CallbackStack, a::Real, b::DataVariable) = mult(a, b)

function mult(stack::CallbackStack, a::Real, b::GradVariable)
    y = GradVariable(mult(a, b.data))
    push!(stack, ReverseScalarMult(y, a, b))
    return y
end

mult(b::Variable, a::Real) = mult(a, b)

mult(stack::CallbackStack, b::Variable, a::Real) = mult(stack, a, b)
