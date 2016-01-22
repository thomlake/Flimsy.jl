
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

function wta(x::AbstractMatrix)
    y = zero(x)
    xmax, imax = findmax(x, 1)
    for i = 1:endof(imax)
        y[imax[i]] = xmax[i]
    end
    return y
end

wta(x::Variable) = DataVariable(wta(x.data))

wta(stack::CallbackStack, x::DataVariable) = wta(x)

function wta(stack::CallbackStack, x::GradVariable)
    y = GradVariable(wta(x.data))
    push!(stack, ReverseWTA(y, x))
    return y
end
