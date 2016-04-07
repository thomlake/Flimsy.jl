
type ReverseRecip{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseRecip)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for i in eachindex(x)
        x.grad[i] -= y.grad[i] / abs2(x.data[i])
    end
    return nothing
end

function recip!(y::AbstractArray, x::AbstractArray)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = 1.0 / x[i]
    end
    return y
end

recip(x::AbstractArray) = recip!(similar(x), x)

recip(scope::Scope, x::Variable) = DataVariable(recip!(similar(scope, x.data), x.data))

function recip(scope::GradScope, x::GradVariable)
    y = GradVariable(recip!(similar(scope, x.data), x.data), similar(scope, x.data, 0))
    push_callback!(scope, ReverseRecip(y, x))
    return y
end
