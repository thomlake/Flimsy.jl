module Flimsy

using Distributions
using HDF5
using JSON
import StatsBase: softmax, softmax!

include("configure.jl")

abstract Component{T}
abstract ReverseOperation

# Distributions exports
export Normal,
       Uniform

export Component,
       ReverseOperation

export @flimsy_inbounds

export @component

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
       GradScope,
       DynamicScope,
       DynamicGradScope,
       StaticScope,
       StaticGradScope,
       CallbackStack,
       backprop!,
       gradient!,
       reset!,
       allocate,
       available,
       vartype

export Sequence

export glorot,
       orthonormal

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

export Runtime, 
       setup

export Components

export Patience

const FLIMSY_DEFAULT_HEAP_SIZE = 1_073_741_824

include("flimsy_macros.jl")
include("component_macro.jl")
include("variable.jl")
include("argmax.jl")
include("scope.jl")
include("sequence.jl")
include("ops.jl")
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

end