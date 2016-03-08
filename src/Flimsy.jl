module Flimsy

using Distributions
using HDF5

abstract Component{T}
abstract ReverseOperation

# Distributions exports
export Normal,
       Uniform

export Component,
       ReverseOperation

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

export getparams,
       getnamedparams

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
       AdaDelta,
       Adam,
       update!,
       optimizer

export GradientNoise

export check_gradients

export Cost

export Model

export Components

export ExternalEvaluation,
       FunctionEvaluation,
       NoImprovement,
       IsEqual,
       Patience,
       Progress,
       converged,
       timer_start,
       timer_stop,
       evaluate,
       epoch,
       best

const FLIMSY_DEFAULT_HEAP_SIZE = 1_073_741_824

include("component_macro.jl")
include("variable.jl")
include("argmax.jl")
include("getparams.jl")
include("scope.jl")
include("sequence.jl")
include("ops.jl")
include("initialization.jl")
include("gradient_noise.jl")
include("model.jl")
include("check_gradients.jl")
include("ctc.jl")
include("cost.jl")
include("fit.jl")
include("components.jl")
include("extras.jl")
include("progress.jl")
include("io.jl")
include("inplace.jl")

end