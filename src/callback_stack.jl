
typealias CallbackStack Array{ReverseOperation,1}

function backprop!(stack::CallbackStack)
    for i = endof(stack):-1:1
        stack[i]()
    end
    empty!(stack)
end

# function gradient!(f::Function, params::Component, args...)
#     stack = CallbackStack()
#     y = f(stack, params, args...)
#     backprop!(stack)
#     return y
# end

function gradient!(f::Function, args...)
    stack = CallbackStack()
    y = f(stack, args...)
    backprop!(stack)
    return y
end
