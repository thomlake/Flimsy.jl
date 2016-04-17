
type ReverseNorm2 <: ReverseOperation
    y::GradVariable
    x::GradVariable
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

function norm2!(y::Matrix, x::Matrix)
    sumabs2!(y, x)
    @flimsy_inbounds for i in eachindex(y)
        y[i] = sqrt(y[i])
    end
    return y
end

norm2(scope::Scope, x::Variable) = DataVariable(norm2!(Matrix{FloatX}(1, size(x, 2)), x.data))

function norm2(scope::GradScope, x::GradVariable)
    y_data = Matrix{FloatX}(1, size(x, 2))
    y_grad = zero(y_data)
    y = GradVariable(norm2!(y_data, x.data), y_grad)
    push_callback!(scope, ReverseNorm2(y, x))
    return y
end
