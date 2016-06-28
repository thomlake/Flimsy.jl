
abstract Activation

immutable Sigmoid <: Activation end
Base.string(f::Sigmoid) = "Sigmoid"
activate(f::Sigmoid, x) = sigmoid(x)
activate(scope::Scope, f::Sigmoid, x) = sigmoid(scope, x)

immutable Tanh <: Activation end
Base.string(f::Tanh) = "Tanh"
activate(f::Tanh, x) = tanh(x)
activate(scope::Scope, f::Tanh, x) = tanh(scope, x)

immutable Relu <: Activation end
Base.string(f::Relu) = "Relu"
activate(f::Relu, x) = relu(x)
activate(scope::Scope, f::Relu, x) = relu(scope, x)

immutable Wta <: Activation end
Base.string(f::Wta) = "Wta"
activate(f::Wta, x) = wta(x)
activate(scope::Scope, f::Wta, x) = wta(scope, x)

immutable Softmax <: Activation end
Base.string(f::Softmax) = "Softmax"
activate(f::Softmax, x) = softmax(x)
activate(scope::Scope, f::Softmax, x) = softmax(scope, x)

immutable Identity <: Activation end
Base.string(f::Identity) = "Identity"
activate(f::Identity, x) = x
activate(scope::Scope, f::Identity, x) = x

const ACTIVATION_LOOKUP = Dict{ASCIIString,DataType}(
    "Sigmoid" => Sigmoid,
    "Tanh"    => Tanh,
    "Relu"    => Relu,
    "Wta"     => Wta,
    "Softmax" => Softmax,
    "Identity" => Identity,
)