# Flimsy.jl
Flimsy.jl is a Julia package for expressing and training a rich class of machine learning models whose parameters are differentiable with respect to a scalar loss function, i.e., neural networks.

Computations are described natively within Julia, making it _relatively_ easy to express complicated models. 
For example, models with [attentional components](http://arxiv.org/abs/1409.0473) or arbitrary size persistent [memory structures](http://arxiv.org/abs/1503.08895) can be easily expressed using Flimsy.jl 

## Features
- Automatic gradient computation.
- Express computation with native control flow structures (`for`, `while`, recursion, etc).
- Extensible.
- Group parameters and functionality via components.
- No compilation process (outside of Julia's own JIT compiler).

## Un-Features
- No GPU support.
- No automatic computation graph optimization.

## Why?
Flimsy.jl is primarily an experiment in interface design for neural network centric machine learning libraries. 
It aims to overcome what I see as the biggest interface drawbacks of popular libraries such as 
[Theano](http://deeplearning.net/software/theano/), 
[Torch](http://torch.ch/), and 
[TensorFlow](https://www.tensorflow.org/).<br>
**N.B.** I'm much less familiar with the other libararies listed above than Theano, which I have worked with extensively for several years. As such the below criticism may or may not be applicable to all the libraries listed above.

- **Awkward, limited, and non-native control flow structures**<br>
Flimsy.jl sidesteps the need for an explict computational graph structure and the *special* control flow functions they bring with them, e.g., `scan`, by implicitly constructing a backward graph at runtime. This is done by pushing a closure (implemented as a type with a `call` method) onto a shared stack after each function application. Popping and calling the closures until the stack is empty implicitly carries out backpropagation.

- **Lack of reuseable and composable sub-components**<br>
Flimsy.jl defines an abstract `Component` type for coupling parameters and functionality. Using Julia's multiple dispatch and a common set of component functions names allows the creation of a library of `Components` which can easily be composed to form larger `Components`. See [`examples/rnn_comparison.jl`](https://github.com/thomlake/Flimsy.jl/blob/master/examples/rnn_comparison.jl) for a practical example.

- **Two Language Syndrome**<br>
Flimsy.jl is written entirely in Julia, and Julia is fast. 
This means new primitive operations can be defined without switching languages and writing wrappers.

Unfortunately Flimsy.jl is currently nowhere near performant enough to serve as 
a substitute for the libraries mentioned above in many cases. 

## Disclaimer
Flimsy.jl is experimental software. 
There are probably bugs to be fixed and certainly optimizations to be made. Any potential user would be well advised to make liberal use of the `check_gradients` function to test that gradients are being calculated correctly for their particular model.

## Installation
Flimsy.jl is currently unregistered, to install use
```julia
julia> Pkg.clone("https://github.com/thomlake/Flimsy.jl.git")
```

## Example
An example Logistic Regression implementation is given below. 
This and several other builtin components are available in the `Flimsy.Components` module.

```julia
using Flimsy
using Synthetic # Data generation: https://github.com/thomlake/Synthetic.jl

# Parameter definition
immutable Params{V<:Variable} <: Component{V}
    w::V
    b::V
end

# Default constructor
Params(m, n) = Params(w=randn(m, n), b=zeros(m, 1))

# Computation the model performs
@comp score(θ::Params, x::Variable) = affine(θ.w, x, θ.b)
@comp predict(θ::Params, x::Variable) = argmax(score(θ, x))
@comp probs(θ::Params, x::Variable) = softmax(score(θ, x))
@comp cost(θ::Params, x::Variable, y) = Cost.categorical_cross_entropy_with_scores(score(θ, x), y)

# Check gradients using finite differences
function check()
    println("checking gradients....")
    n_features, n_classes, n_samples = 20, 3, 5
    data = rand(Synthetic.MixtureTask(n_features, n_classes), n_samples)
    X = hcat(map(first, data)...)
    y = vcat(map(last, data)...)
    θ = Runtime(Params(n_classes, n_features))
    check_gradients(cost, θ, Input(X), y)
end

# Train/Test
function main()
    srand(sum(map(Int, collect("Flimsy"))))
    n_features, n_classes = 20, 3
    n_train, n_test = 50, 50
    D = Synthetic.MixtureTask(n_features, n_classes)
    data_train = rand(D, n_train)
    data_test = rand(D, n_test)
    X_train, y_train = hcat(map(first, data_train)...), vcat(map(last, data_train)...)
    X_test, y_test = hcat(map(first, data_test)...), vcat(map(last, data_test)...)

    θ = Runtime(Params(n_classes, n_features))
    opt = optimizer(RmsProp, θ, learning_rate=0.01, decay=0.9)
    start_time = time()
    for i = 1:100
        nll = cost(θ, Input(X_train), y_train; grad=true)
        update!(opt)
        i % 10 == 0 && println("epoch => $i, nll => $nll")
    end
    println("wall time   => ", time() - start_time)
    println("train error => ", sum(y_train .!= predict(θ, Input(X_train))) / n_train)
    println("test error  => ", sum(y_test .!= predict(θ, Input(X_test))) / n_test)
end

("-c" in ARGS || "--check" in ARGS) && check()
main()
```

## Directory Layout
### `src/`
Core module functionality.

### `test/`
Unit tests (uses [FactCheck.jl](https://github.com/JuliaLang/FactCheck.jl)).

### `src/ops/`
Differentiable operations.

### `src/cost/`
Module containing common cost functions.

### `src/components/`
Module containing common ML models.

### `src/ctc.jl`
Module containing utility functions for Connectionist Temporal Classification cost function.

### `src/extras/`
Module containing several utility functions.

