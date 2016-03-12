
type ReverseRelu{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
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


relu(scope::Scope, x::Variable) = DataVariable(relu!(similar(scope, x.data), x.data))

function relu(scope::GradScope, x::GradVariable)
    y = GradVariable(relu!(similar(scope, x.data), x.data), similar(scope, x.data, 0))
    push_callback!(scope, ReverseRelu(y, x))
    return y
end
