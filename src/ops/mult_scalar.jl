
type ReverseScalarMult{F<:Real} <: ReverseOperation
    c::GradVariable
    a::F
    b::GradVariable
end

function call(rop::ReverseScalarMult)
    c = rop.c
    a = rop.a
    b = rop.b
    @flimsy_inbounds for i in eachindex(c)
        b.grad[i] += a * c.grad[i]
    end
    return nothing
end

function mult!(c::AbstractArray, a::Real, b::AbstractArray)
    @flimsy_inbounds for i in eachindex(b)
        c[i] = a * b[i]
    end
    return c
end

mult(a::Real, b::AbstractArray) = a .* b

mult(scope::Scope, a::Real, b::Variable) = DataVariable(mult!(similar(b.data), a, b.data))

function mult(scope::GradScope, a::Real, b::GradVariable)
    c = GradVariable(mult!(similar(b.data), a, b.data), zero(b.data))
    push_callback!(scope, ReverseScalarMult(c, a, b))
    return c
end

mult(scope::Scope, b::Variable, a::Real) = mult(scope, a, b)
