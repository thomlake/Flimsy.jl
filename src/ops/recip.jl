
type ReverseRecip <: ReverseOperation
    y::Variable
    x::Variable
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

recip(scope::Scope, x::AbstractValue) = Constant(recip!(similar(x.data), x.data))

function recip(scope::GradScope, x::Variable)
    y = Variable(recip!(similar(x.data), x.data), zero(x.data))
    push_callback!(scope, ReverseRecip(y, x))
    return y
end
