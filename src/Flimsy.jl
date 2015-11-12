module Flimsy

export AbstractVariable,
       Variable,
       NativeVariable,
       BPStack,
       Component,
       backprop!,
       gradient!,
       getparams,
       getnamedparams,
       @flimsy

export sigmoid,
       relu,
       softmax,
       wta,
       linear,
       minus,
       concat,
       affine,
       dropout!

export Gaussian,
       Uniform,
       Glorot,
       Orthonormal,
       Sparse,
       Identity,
       Zeros

export SGD,
       Momentum,
       Nesterov,
       RMSProp,
       AdaDelta,
       update!,
       optimizer

export gradcheck

include("var.jl")
include("core.jl")
include("macros.jl")
include("hashmat.jl")
include("ops.jl")
include("initialization.jl")
include("fit.jl")
include("gradcheck.jl")

module Cost
    using ..Flimsy
    include("cost.jl")
    module CTC
        using ....Flimsy
        include("ctc.jl")
    end
end

module Extras
    using ..Flimsy
    include("extras.jl")
    include("progress.jl")
end

module Components
    using ..Flimsy
    import StatsBase: predict
    import Distributions: probs

    export LogisticRegression,
           LinearRegression,
           CTCOutput,
           FeedForwardLayer,
           LayerStack,
           Recurrent,
           GatedRecurrent,
           LSTM,
           score,
           probs,
           predict,
           feedforward,
           step,
           unfold

    include("components/logistic_regression.jl")
    include("components/linear_regression.jl")
    include("components/ctcoutput.jl")
    include("components/feedforwardlayer.jl")
    include("components/recurrent.jl")
    include("components/gatedrecurrent.jl")
    include("components/lstm.jl")
end

module SampleData
    using ..Flimsy
    include("sampledata.jl")
end

end #module Flimsy
