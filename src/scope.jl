
typealias CallbackStack Array{ReverseOperation,1}

abstract Scope

type DataScope <: Scope end

type GradScope <: Scope
    stack::CallbackStack
end

GradScope() = GradScope(CallbackStack())

function backprop!(scope::GradScope)
    stack = scope.stack
    while length(stack) > 0
        pop!(stack)()
    end
end

function push_callback!(scope::GradScope, cb::ReverseOperation)
    push!(scope.stack, cb)
end
