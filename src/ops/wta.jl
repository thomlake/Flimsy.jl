
type ReverseWta <: ReverseOperation
    y::Variable
    x::Variable
end

function call(rop::ReverseWta)
    y = rop.y
    x = rop.x
    _, imax = findmax(x.data, 1)
    @flimsy_inbounds for i = 1:endof(imax)
        x.grad[imax[i]] += y.grad[imax[i]]
    end
    return nothing
end

function wta!(y::AbstractMatrix, x::AbstractMatrix)
    xmax, imax = findmax(x, 1)
    @flimsy_inbounds for i = 1:endof(imax)
        y[imax[i]] = xmax[i]
    end
    return y
end

wta(x::AbstractMatrix) = wta!(zero(x), x)

wta(scope::Scope, x::AbstractValue) = Constant(wta!(zero(x.data), x.data))

function wta(scope::GradScope, x::Variable)
    y = Variable(wta!(zero(x.data), x.data), zero(x.data))
    push_callback!(scope, ReverseWta(y, x))
    return y
end
