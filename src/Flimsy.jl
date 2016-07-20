module Flimsy

using Distributions
using HDF5
using JSON
import StatsBase: softmax, softmax!

typealias FloatX Float32

abstract Component
abstract ReverseOperation

# Distributions exports
export Normal,
       Uniform

export Component,
       ReverseOperation,
       FloatX

export @flimsy_inbounds

export @with

export @backprop,
       @run

export AbstractValue,
       Constant,
       Variable,
       Input,
       is_matrix,
       is_column_vector,
       is_row_vector,
       is_scalar

export argmax,
       argmaxneq

export Scope,
       RunScope,
       GradScope,
       BackpropStack,
       backprop!,
       computing_grads

export Sequence,
       NTupleSequence

export Activation,
       Sigmoid,
       Tanh,
       Relu,
       Wta,
       Softmax,
       Identity,
       activate

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

# export shuffled

include("configure.jl")
include("flimsy_macros.jl")
include("inplace.jl")
include("scope.jl")
include("variable.jl")
include("ops.jl")
include("ctc.jl")
include("cost/cost.jl")

include("component_macro.jl")
include("graph_macros.jl")
include("argmax.jl")
include("sequence.jl")
include("activation.jl")
include("initialization.jl")
include("gradient_noise.jl")
# # include("runtime.jl")
include("conversion.jl")
include("check_gradients.jl")
include("fit.jl")
include("components/components.jl")
include("extras/extras.jl")
include("progress.jl")
include("io.jl")
# # include("shufflediter.jl")

end