
abstract Activation

immutable Sigmoid <: Activation end
Base.string(f::Sigmoid) = "Sigmoid"
call(f::Sigmoid, x) = sigmoid(x)
call(f::Sigmoid, scope::Scope, x) = sigmoid(scope, x)

immutable Tanh <: Activation end
Base.string(f::Tanh) = "Tanh"
call(f::Tanh, x) = tanh(x)
call(f::Tanh, scope::Scope, x) = tanh(scope, x)

immutable Relu <: Activation end
Base.string(f::Relu) = "Relu"
call(f::Relu, x) = relu(x)
call(f::Relu, scope::Scope, x) = relu(scope, x)

immutable Wta <: Activation end
Base.string(f::Wta) = "Wta"
call(f::Wta, x) = wta(x)
call(f::Wta, scope::Scope, x) = wta(scope, x)

immutable Softmax <: Activation end
Base.string(f::Softmax) = "Softmax"
call(f::Softmax, x) = softmax(x)
call(f::Softmax, scope::Scope, x) = softmax(scope, x)

const ACTIVATION_LOOKUP = Dict{ASCIIString,DataType}(
    "Sigmoid" => Sigmoid,
    "Tanh"    => Tanh,
    "Relu"    => Relu,
    "Wta"     => Wta,
    "Softmax" => Softmax,
)