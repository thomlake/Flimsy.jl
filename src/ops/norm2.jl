
type ReverseNorm2{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseNorm2)
    y = rop.y
    x = rop.x
    @flimsy_inbounds for j = 1:size(x, 2)
        for i = 1:size(x, 1)
            x.grad[i,j] += y.grad[1,j] * x.data[i,j] / y.data[1,j]
        end
    end
    return nothing
end

function norm2!{T}(y::Matrix{T}, x::Matrix{T})
    sumabs2!(y, x)
    @flimsy_inbounds for i in eachindex(y)
        y[i] = sqrt(y[i])
    end
    return y
end

norm2(scope::Scope, x::Variable) = DataVariable(norm2!(allocate(scope, eltype(x.data), (1, size(x, 2))), x.data))

function norm2(scope::GradScope, x::GradVariable)
    y_data = allocate(scope, eltype(x.data), (1, size(x, 2)))
    y = GradVariable(norm2!(y_data, x.data), similar(scope, y_data, 0))
    push_callback!(scope, ReverseNorm2(y, x))
    return y
end
