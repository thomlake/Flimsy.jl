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
       GradInput,
       DataVariable,
       Input,
       is_matrix,
       is_column_vector,
       is_row_vector,
       is_scalar

export getparams,
       getnamedparams

export CallbackStack,
       push_callback!,
       backprop!,
       gradient!

export check_gradients

export Cost

export Components

export compile

abstract Component{T}
abstract ReverseOperation

include("component_macro.jl")
include("variable.jl")
include("getparams.jl")
include("callback_stack.jl")
include("ops.jl")
include("check_gradients.jl")
include("cost.jl")
include("components.jl")
include("compile.jl")
include("demo.jl")



end