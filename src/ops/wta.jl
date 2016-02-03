
type ReverseWTA{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

# call{T<:DataVariable}(rop::ReverseWTA{T}) = nothing

function call(rop::ReverseWTA)
    y = rop.y
    x = rop.x
    _, imax = findmax(x.data, 1)
    for i = 1:endof(imax)
        x.grad[imax[i]] += y.grad[imax[i]]
    end
    return nothing
end

function wta!(y::AbstractMatrix, x::AbstractMatrix)
    xmax, imax = findmax(x, 1)
    for i = 1:endof(imax)
        y[imax[i]] = xmax[i]
    end
    return y
end

wta(x::AbstractMatrix) = wta!(zero(x), x)

wta(scope::Scope, x::Variable) = DataVariable(wta!(similar(scope, x.data, 0), x.data))

function wta(scope::GradScope, x::GradVariable)
    y = GradVariable(wta!(similar(scope, x.data, 0), x.data), similar(scope, x.data, 0))
    push_callback!(scope, ReverseWTA(y, x))
    return y
end
