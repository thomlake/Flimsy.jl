
# minus(Real, Variable) 
type ReverseScalarMinusMatrix <: ReverseOperation
    y::GradVariable
    x::GradVariable
end

function call(rop::ReverseScalarMinusMatrix)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for i in eachindex(y)
        x.grad[i] -= y.grad[i]
    end
    return nothing
end

function minus!(y::AbstractArray, a::Real, x::AbstractArray)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = a - x[i]
    end
    return y
end

minus(scope::Scope, a::Real, x::Variable) = DataVariable(minus!(similar(x.data), a, x.data))

function minus(scope::GradScope, a::Real, x::GradVariable)
    y = GradVariable(minus!(similar(x.data), a, x.data), zero(x.data))
    push_callback!(scope, ReverseScalarMinusMatrix(y, x))
    return y
end

# minus(Variable, Real)
type ReverseMatrixMinusScalar <: ReverseOperation
    y::GradVariable
    x::GradVariable
end

function call(rop::ReverseMatrixMinusScalar)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for i in eachindex(y)
        x.grad[i] += y.grad[i]
    end
    return nothing
end

function minus!(y::AbstractArray, x::AbstractArray, a::Real)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = x[i] - a
    end
    return y
end

minus(scope::Scope, x::Variable, a::Real) = DataVariable(minus!(similar(x.data), x.data, a))

function minus(scope::GradScope, x::GradVariable, a::Real)
    y = GradVariable(minus!(similar(x.data), x.data, a), zero(x.data))
    push_callback!(scope, ReverseMatrixMinusScalar(y, x))
    return y
end
