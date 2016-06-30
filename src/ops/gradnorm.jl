
type ReverseGradNorm{F<:AbstractFloat} <: ReverseOperation
    s::F
    x::GradVariable
end

function call(rop::ReverseGradNorm)
    s = rop.s
    dx = rop.x.grad
    @flimsy_inbounds for j = 1:size(dx, 2)
        ss = 0.0
        for i = 1:size(dx, 1)
            ss += abs2(dx[i,j])
        end
        c = s / sqrt(ss)
        for i = 1:size(dx, 1)
            dx[i,j] *= c
        end
    end
    return nothing
end

gradnorm(scope::Scope, x::Variable, s::AbstractFloat=1.0) = x

function gradnorm(scope::GradScope, x::GradVariable, s::AbstractFloat=1.0)
    push_callback!(scope, ReverseGradNorm(s, x))
    return x
end
