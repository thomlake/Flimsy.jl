module Flimsy

export Variable,
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
       decat,
       affine,
       dropout!,
       threshold

export Gaussian,
       Uniform,
       Glorot,
       Orthonormal,
       Sparse,
       Identity,
       Zeros

export GradientDescent,
       Momentum,
       Nesterov,
       RMSProp,
       AdaDelta,
       Adam,
       update!,
       optimizer

export GradientNoise,
       gradient_noise_scalar

export gradcheck

export Progress

include("var.jl")
include("core.jl")
include("macros.jl")
include("hashmat.jl")
include("ops.jl")
include("initialization.jl")
include("fit.jl")
include("gradientnoise.jl")
include("gradcheck.jl")
include("progress.jl")

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
end

module Components
    using ..Flimsy
    import StatsBase: predict
    import Distributions: probs

    abstract RecurrentComponent{T,M,N} <: Component

    export LogisticRegression,
           LinearRegression,
           MultilabelClassifier,
           LinearSVM,
           CTCOutput,
           FeedForwardLayer,
           LayerStack,
           RecurrentComponent,
           SimpleRecurrent,
           GatedRecurrent,
           ResidualRecurrent,
           LSTM,
           cost,
           score,
           probs,
           predict,
           feedforward,
           step,
           unfold

    include("components/logistic_regression.jl")
    include("components/linear_regression.jl")
    include("components/multilabel_classifier.jl")
    include("components/linear_svm.jl")
    include("components/ctcoutput.jl")
    include("components/feedforwardlayer.jl")
    include("components/simple_recurrent.jl")
    include("components/gatedrecurrent.jl")
    include("components/lstm.jl")
    include("components/residualrecurrent.jl")
end

module Demo
    using ..Flimsy
    include("demo/xor.jl")
    include("demo/addtask.jl")
    include("demo/mog.jl")
end


end #module Flimsy
