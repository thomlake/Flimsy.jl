
type ReverseDecat{T<:GradVariable} <: ReverseOperation
    ys::Vector{T}
    x::T
end

function call(rop::ReverseDecat)
    ys = rop.ys
    x = rop.x
    m, n = size(x)
    for j = 1:n
        for i = 1:m
            x.grad[i,j] += ys[i].grad[j]
        end
    end
    return nothing
end

decat{V<:Variable}(x::V) = [V(x.data[i,:]) for i = 1:size(x, 1)]

function decat(stack::CallbackStack, x::GradVariable)
    ys = decat(x)
    push_callback!(stack, ReverseDecat(ys, x))
    return ys
end
