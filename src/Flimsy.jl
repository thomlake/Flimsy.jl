module Flimsy

using Distributions
using HDF5
using JSON
import StatsBase: softmax, softmax!

typealias FloatX Float32

include("configure.jl")

abstract Component{T}
abstract ReverseOperation

# Distributions exports
export Normal,
       Uniform

export Component,
       ReverseOperation,
       FloatX

export @flimsy_inbounds

export @comp

export Variable,
       GradVariable,
       DataVariable,
       Input,
       is_matrix,
       is_column_vector,
       is_row_vector,
       is_scalar

export argmax,
       argmaxneq

export Scope,
       DataScope,
       GradScope,
       CallbackStack,
       backprop!

export Sequence

export Activation,
       Sigmoid,
       Tanh,
       Relu,
       Wta

export glorot,
       orthonormal,
       spectral

export GradientDescent,
       Momentum,
       Nesterov,
       RmsProp,
       Graves,
       AdaDelta,
       Adam,
       update!,
       optimizer

export GradientNoise

export check_gradients

export Cost

export Runtime

export Components

export Patience

export ShuffledIter

include("flimsy_macros.jl")
include("component_macro.jl")
include("variable.jl")
include("argmax.jl")
include("scope.jl")
include("sequence.jl")
include("ops.jl")
include("activation.jl")
include("initialization.jl")
include("gradient_noise.jl")
include("runtime.jl")
include("conversion.jl")
include("check_gradients.jl")
include("ctc.jl")
include("cost/cost.jl")
include("fit.jl")
include("components/components.jl")
include("extras/extras.jl")
include("progress.jl")
include("io.jl")
include("inplace.jl")
include("shufflediter.jl")

end