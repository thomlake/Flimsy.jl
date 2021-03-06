
type ReverseSigmoid <: ReverseOperation
    y::GradVariable
    x::GradVariable
end

function call(rop::ReverseSigmoid)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for i in eachindex(x)
        x.grad[i] += y.data[i] * (1 - y.data[i]) * y.grad[i]
    end
    return nothing
end

function sigmoid!(y::AbstractArray, x::AbstractArray)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    return y
end

function sigmoid(x::AbstractArray)
    y = zero(x)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    return y
end

sigmoid(scope::Scope, x::Variable) = DataVariable(sigmoid!(similar(x.data), x.data))

function sigmoid(scope::GradScope, x::GradVariable)
    y = GradVariable(sigmoid!(similar(x.data), x.data), zero(x.data))
    push_callback!(scope, ReverseSigmoid(y, x))
    return y
end
