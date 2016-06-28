
module Components

using .. Flimsy
import StatsBase: predict
using HDF5: HDF5Group, attrs

export score,
       predict,
       probs,
       unfold,
       cost,
       weighted_cost,
       feedforward,
       initial_state

export RecurrentComponent,
       RecurrentComponent1,
       RecurrentComponent2

export ValueComponent,
       VectorComponent,
       EmptyComponent

export LinearModel,
       LinearRegression,
       SoftmaxRegression,
       SigmoidRegression,
       CtcOutput,
       FeedForward,
       Fusion,
       SimpleRecurrent,
       GatedRecurrent,
       Lstm

abstract RecurrentComponent <: Component
abstract RecurrentComponent1 <: RecurrentComponent
abstract RecurrentComponent2 <: RecurrentComponent

include("constructor.jl")
include("common.jl")
include("value_component.jl")
include("linear_model.jl")
# include("ctc_output.jl")
# include("feedforward.jl")
# include("fusion.jl")
# include("simple_recurrent.jl")
# include("gated_recurrent.jl")
# include("lstm.jl")

end