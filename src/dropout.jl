
function dropout!(scope::Scope, x::Variable, p::AbstractFloat)
    for i in eachindex(x)
        x.data[i] *= 1 - p
    end
    return x
end

function dropout!(scope::GradScope, x::Variable, p::AbstractFloat)
    for i in eachindex(x)
        if rand() < p
            x.data[i] = 0
        end
    end
    return x
end

function dropout!(scope::GradScope, x::Variable, p::Matrix)
    
    for i in eachindex(x)
        if p[i]
            x.data[i] = 0
        end
    end
    return x
end

type ReverseDropout{T<:GradVariable} <: ReverseOperation
    x::T
    m::BitMatrix
end

function call(rop::ReverseDropout)
    x = rop.x
    m = rop.m
    for i in eachindex(x)
        if m[i]
            x.grad[i] = 0
        end
    end
    return nothing
end

function dropout_adjust!(x::AbstractArray, p::AbstractFloat)
    for i in eachindex(x)
        x.data[i] *= 1 - p
    end
    return x
end

function dropout!(x::AbstractArray, m::BitMatrix)
    for i in eachindex(x)
        if m[i]
            x[i] = 0
        end
    end
    return x
end

function dropout!(scope::Scope, x::Variable, p::AbstractFloat)
    dropout_adjust!(x.data, p)
    return x
end

function dropout!(scope::GradScope, x::GradVariable, m::BitMatrix)
    size(x) == size(m) || throw(DimensionMismatch("dropout mask with size $(size(m)) cannot be applied to variable with size $(size(x))"))
    m = rand(size(x)) .< m
    dropout!(x.data, m)
    push_callback!(scope, ReverseDropout(x, m))
    return x
end

function dropout!(scope::GradScope, x::GradVariable, p::AbstractFloat)
    m = rand(size(x)) .< m
    return dropout!(scope, x, m)
end
