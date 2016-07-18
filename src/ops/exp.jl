
type ReverseExp <: ReverseOperation
    y::Variable
    x::Variable
end

function call(rop::ReverseExp)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for i in eachindex(x)
        x.grad[i] += y.data[i] * y.grad[i]
    end
    return nothing
end

function exp!(y::AbstractArray, x::AbstractArray)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = exp(x[i])
    end
    return y
end

Base.exp(scope::Scope, x::AbstractValue) = Constant(exp!(similar(x.data), x.data))

function Base.exp(scope::GradScope, x::Variable)
    y = Variable(exp!(similar(x.data), x.data))
    push_callback!(scope, ReverseExp(y, x))
    return y
end
