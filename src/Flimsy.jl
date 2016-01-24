module Flimsy

using Distributions
using FastAnonymous

# Distributions exports
export Normal,
       Uniform

export @anon

export Component,
       GradComponent,
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

export CallbackStack,
       push_callback!,
       backprop!,
       gradient!

export glorot,
       orthonormal

export GradientDescent,
       Momentum,
       Nesterov,
       RMSProp,
       AdaDelta,
       Adam,
       update!,
       optimizer

export check_gradients

export Cost

export Components

export Progress,
       converged,
       stop,
       criteria,
       epoch,
       best

abstract Component{T}

abstract ReverseOperation

include("component_macro.jl")
include("variable.jl")
include("component_constructor.jl")
include("argmax.jl")
include("getparams.jl")
include("callback_stack.jl")
include("ops.jl")
include("initialization.jl")
include("fit.jl")
include("check_gradients.jl")
include("cost.jl")
include("components.jl")
include("extras.jl")
include("progress.jl")
include("demo.jl")



end