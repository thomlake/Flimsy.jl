
type ReverseRelu <: ReverseOperation
    y::Variable
    x::Variable
end

function call(rop::ReverseRelu)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for i in eachindex(x)
        if x.data[i] > 0
            x.grad[i] += y.grad[i]
        end
    end
    return nothing
end

function relu!(y::AbstractArray, x::AbstractArray)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = max(0, x[i])
    end
    return y
end

relu(x::AbstractArray) = max(0, x)


relu(scope::Scope, x::AbstractValue) = Constant(relu!(similar(x.data), x.data))

function relu(scope::GradScope, x::Variable)
    y = Variable(relu!(similar(x.data), x.data), zero(x.data))
    push_callback!(scope, ReverseRelu(y, x))
    return y
end
