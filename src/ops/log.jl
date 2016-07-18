
type ReverseLog <: ReverseOperation
    y::Variable
    x::Variable
end

function call(rop::ReverseLog)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for i in eachindex(x)
        x.grad[i] += y.grad[i] / x.data[i]
    end
    return nothing
end

function log!(y::AbstractArray, x::AbstractArray)
    @flimsy_inbounds for i in eachindex(x)
        # x[i] > 0 || throw(OperationError("log domain error: $(x[i])"))
        y[i] = log(x[i])
    end
    return y
end

Base.log(scope::Scope, x::AbstractValue) = Constant(log!(similar(x.data), x.data))

function Base.log(scope::GradScope, x::Variable)
    y = Variable(log!(similar(x.data), x.data))
    push_callback!(scope, ReverseLog(y, x))
    return y
end
