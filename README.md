# Flimsy.jl
Flimsy.jl is a Julia package for expressing and training a rich class of machine learning models whose parameters are differentiable with respect to a scalar loss function, i.e., neural networks.

Computations are described natively within Julia, making it _relatively_ easy to express complicated models. For example, models with [attentional components](http://arxiv.org/abs/1409.0473) or arbitrary size persistent [memory structures](http://arxiv.org/abs/1503.08895) can be easily expressed using Flimsy.jl (see [attention.jl](https://github.com/thomlake/Flimsy.jl/blob/master/examples/attention.jl) for an example).

## Features
- Automatic gradient computation
- Express computation with native control flow structures (`for`, `while`, recursion, etc)
- Extensible
- Group parameters and functionality via components
- No compilation process (outside of Julia's own JIT compiler)

## Un-Features
- No GPU support
- No automatic computation graph optimization
- No memory pre-allocation or reuse

## Why?
Flimsy.jl is primarily an experiment in interface design for neural network centric machine learning libraries. It aims to overcome what I see as the biggest interface drawbacks of popular libraries such as [Theano](http://deeplearning.net/software/theano/), [Torch](http://torch.ch/), and [TensorFlow](https://www.tensorflow.org/).

- **Awkward, limited, and non-native control flow structures**<br>
Flimsy.jl sidesteps the need for an explict computational graph by pushing a closure onto a shared stack after each function application. Popping and calling the closures until the stack is empty implicitly carries out backpropagation.

- Disconnected model definition and execution phases
- Inability to build reuseable sub-models

Unfortunately Flimsy.jl is currently nowhere near performant enough to serve as a substitute for the libraries mentioned above in most cases. 

## Disclaimer
Flimsy.jl is experimental software. There are probably bugs to be fixed and certainly optimizations to be made. Any potential user would be well advised to make liberal use of the `gradcheck` function to test that gradients are being calculated correctly for their particular model.

## Installation
Flimsy.jl is currently unregistered, to install use
```julia
julia> Pkg.clone("https://github.com/thomlake/Flimsy.jl.git")
```

## Example
An example Logistic Regression implementation is given below. This and several other builtin components are available in the `Flimsy.Components` module.

```julia
using Flimsy

immutable Params{T,M,N} <: Component
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end
Params(m, n) = Params(Variable(randn(m, n)), Variable(zeros(m)))

@flimsy score(θ::Params, x::Variable) = affine(θ.w, x, θ.b)
@flimsy predict(θ::Params, x::Variable) = Flimsy.Extras.argmax(score(θ, x))
@flimsy probs(θ::Params, x::Variable) = softmax(score(θ, x))
@flimsy cost(θ::Params, x::Variable, y) = Flimsy.Cost.cat(y, probs(θ, x))

function run()
    n_classes, n_features = 3, 20
    n_train, n_test = 50, 50
    D = Flimsy.SampleData.MoG(n_classes, n_features)
    X_train, Y_train = rand(D, n_train)
    X_test, Y_test = rand(D, n_test)

    θ = Params(n_classes, n_features)
    opt = optimizer(RMSProp, θ, learning_rate=0.01, decay=0.9)
    for i = 1:100
        nll = gradient!(cost, θ, X_train, Y_train)[1]
        update!(opt, θ)
        i % 10 == 0 && println("epoch => $i, nll => $nll")
    end
    test_error = sum(Y_test .!= predict(θ, X_test)) / n_test
    println("test error => $test_error")
end
```

## Backpropagation Technique
The overall technique for automating backpropagation is essentially the same stack based approach employed by Andrej Karpathy's [recurrentjs](https://github.com/karpathy/recurrentjs). As computation occurs, Flimsy.jl tracks the application of functions. Each functions internally handle how its application changes backpropagation by pushing closures onto a stack.
