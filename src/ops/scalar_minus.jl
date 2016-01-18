type ReverseScalarMinusMatrix{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseScalarMinusMatrix)
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] -= y.grad[i]
    end
    return nothing
end

minus{V<:Variable}(a::Real, x::V) = V(a .- x.data)

function minus(stack::CallbackStack, a::Real, x::GradVariable)
    y = minus(a, x)
    push_callback!(stack, ReverseScalarMinusMatrix(y, x))
    return y
end

type ReverseMatrixMinusScalar{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseMatrixMinusScalar)
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] += y.grad[i]
    end
    return nothing
end

minus{V<:Variable}(x::V, a::Real) = V(x.data .- a)

function minus(stack::CallbackStack, x::GradVariable, a::Real)
    y = minus(x, a)
    push_callback!(stack, ReverseMatrixMinusScalar(y, x))
    return y
end
