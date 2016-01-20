
typealias CallbackStack Array{ReverseOperation,1}

function push_callback!(stack::CallbackStack, f::ReverseOperation)
    push!(stack, f)
    return stack
end

function backprop!(stack::CallbackStack)
    for i = endof(stack):-1:1
        stack[i]()
    end
    empty!(stack)
end

function gradient!(f::Function, theta::Component, args...)
    stack = CallbackStack()
    y = f(stack, theta, args...)
    backprop!(stack)
    return y
end

function gradient!(f::Function, args...)
    stack = CallbackStack()
    y = f(stack, args...)
    backprop!(stack)
    return y
end