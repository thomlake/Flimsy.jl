
# function dropout!(scope::Scope, x::Variable, p::AbstractFloat)
#     for i in eachindex(x)
#         x.data[i] *= 1 - p
#     end
#     return x
# end

# function dropout!(scope::GradScope, x::Variable, p::AbstractFloat)
#     for i in eachindex(x)
#         if rand() < p
#             x.data[i] = 0
#         end
#     end
#     return x
# end

# function dropout!(scope::GradScope, x::Variable, p::Matrix)
    
#     for i in eachindex(x)
#         if p[i]
#             x.data[i] = 0
#         end
#     end
#     return x
# end

type ReverseDropout <: ReverseOperation
    y::Variable
    x::Variable
    m::BitMatrix
end

function call(rop::ReverseDropout)
    y = rop.y
    x = rop.x
    m = rop.m
    for i in eachindex(x)
        if m[i]
            x.grad[i] = 0
        else:
            x.grad[i] += y.grad[i]
        end
    end
    return nothing
end

function dropout_adjust!(y::AbstractArray, x::AbstractArray, p::AbstractFloat)
    s = 1 - p
    @flimsy_inbounds for i in eachindex(x)
        y[i] = s * x[i] 
    end
    return y
end

function dropout!(y::AbstractArray, x::AbstractArray, m::BitMatrix)
    @assert size(y) == size(x)
    @flimsy_inbounds for i in eachindex(x)
        y[i] = m[i] ? 0 : x[i]
    end
    return y
end

dropout(scope::Scope, x::AbstractValue, p::AbstractFloat) = DataVariable(dropout_adjust!(similar(x.data), x.data))

function dropout(scope::GradScope, x::Variable, m::BitMatrix)
    size(x) == size(m) || throw(DimensionMismatch("dropout mask with size $(size(m)) cannot be applied to variable with size $(size(x))"))
    y = Variable(dropout!(similar(x.data), x.data, m))
    push_callback!(scope, ReverseDropout(y, x, m))
    return y
end

function dropout(scope::GradScope, x::Variable, p::AbstractFloat)
    m = rand(size(x)) .< p
    return dropout(scope, x, m)
end
