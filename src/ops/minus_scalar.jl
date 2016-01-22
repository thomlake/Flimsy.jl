
# minus(Real, Variable) 
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

minus(a::Real, x::AbstractArray) = a .- x

minus(a::Real, x::Variable) = DataVariable(minus(a, x.data))

minus(stack::CallbackStack, a::Real, x::DataVariable) = DataVariable(minus(a, x.data))

function minus(stack::CallbackStack, a::Real, x::GradVariable)
    y = GradVariable(minus(a, x.data))
    push!(stack, ReverseScalarMinusMatrix(y, x))
    return y
end

# minus(Variable, Real)
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

minus(x::AbstractArray, a::Real) = x .- a

minus(x::Variable, a::Real) = DataVariable(minus(x.data, a))

minus(stack::CallbackStack, x::DataVariable, a::Real) = DataVariable(minus(x.data, a))

function minus(stack::CallbackStack, x::GradVariable, a::Real)
    y = GradVariable(minus(x.data, a))
    push!(stack, ReverseMatrixMinusScalar(y, x))
    return y
end
