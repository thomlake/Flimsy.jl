
typealias CallbackStack Array{ReverseOperation,1}

abstract Scope

immutable DataScope <: Scope end

immutable GradScope <: Scope
    stack::CallbackStack
end

GradScope() = GradScope(CallbackStack())

function backprop!(scope::GradScope)
    while length(scope.stack) > 0
        pop!(scope.stack)()
    end
end

function push_callback!(scope::GradScope, cb::ReverseOperation)
    push!(scope.stack, cb)
end

computing_grads() = false

computing_grads(scope) = false

computing_grads(scope::GradScope) = true
