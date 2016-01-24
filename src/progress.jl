
abstract AbstractEvaluation

abstract AbstractExternalEvaluation <: AbstractEvaluation

type ExternalEvaluation <: AbstractExternalEvaluation
    minimize::Bool
    tol::Number
    precision::Int
    curr::Number
    best::Number
    improved::Bool
end

function ExternalEvaluation(; minimize::Bool=true, tol::Number=1e-3, precision::Int=3)
    initial = minimize ? Inf : -Inf
    return ExternalEvaluation(minimize, tol, precision, initial, initial, false)
end

function call(self::ExternalEvaluation, curr::Number)
    self.curr = curr
    if self.minimize
        self.improved = self.best - self.curr > self.tol
        self.best = min(self.best, self.curr)
    else
        self.improved = self.best - self.curr < self.tol
        self.best = max(self.best, self.curr)
    end
    return self.improved
end

abstract AbstractFunctionEvaluation <: AbstractEvaluation

type FunctionEvaluation <: AbstractFunctionEvaluation
    f::Function
    minimize::Bool
    tol::Number
    precision::Int
    curr::Number
    best::Number
    improved::Bool
end

function FunctionEvaluation(f::Function; minimize::Bool=true, tol::Number=1e-3, precision::Int=3)
    initial = minimize ? Inf : -Inf
    return FunctionEvaluation(f, minimize, tol, precision, initial, initial, false)
end

function call(self::FunctionEvaluation)
    self.curr = self.f()
    if self.minimize
        self.improved = self.best - self.curr > self.tol
        self.best = min(self.best, self.curr)
    else
        self.improved = self.best - self.curr < self.tol
        self.best = max(self.best, self.curr)
    end
    return self.improved
end

function Base.show(io::IO, self::AbstractEvaluation)
    curr = round(self.curr, self.precision)
    best = round(self.best, self.precision)
    print(io, "curr => $curr, best => $best")
end


abstract AbstractStoppingCriteria

type NoImprovement <: AbstractStoppingCriteria 
    improved::Bool
end

NoImprovement() = NoImprovement(true)

function call(self::NoImprovement, improved::Bool, curr::Real)
    self.improved = improved
    return !improved
end

function Base.show(io::IO, self::NoImprovement)
    print(io, "improved => ", self.improved)
end


type IsEqual <: AbstractStoppingCriteria
    value::Real
    curr::Real
end

IsEqual(value::Real) = IsEqual(value, NaN)

function call(self::IsEqual, improved::Bool, curr::Real)
    self.curr = curr
    return isequal(self.value, self.curr)
end

function Base.show(io::IO, self::IsEqual)
    sym = isequal(self.value, self.curr) ? " == " : " != "
    print(io, "isequal => ", self.curr, sym, self.value)
end

type Patience{T<:Real} <: AbstractStoppingCriteria
    patience::T
    frustration::Int
    frustration_pad::Int
end

function Patience(patience, frustration)
    if isfinite(patience)
        return Patience(patience, frustration, ceil(Int, log10(patience + 1)))
    else
        return Patience(patience, frustration, 0)
    end
end

Patience(patience::Number=1) = Patience(patience, 0)

function call(self::Patience, improved::Bool, curr::Real)
    if improved
        self.frustration = 0
    else
        self.frustration += 1
    end
    return self.frustration > self.patience
end

function Base.show(io::IO, self::Patience)
    frustration = lpad(self.frustration, self.frustration_pad, " ")
    patience = lpad(self.patience, self.frustration_pad, " ")
    print(io, "patience => ($frustration of $patience)")
end

type Timer
    start_time::Float64
    stop_time::Float64
end

Timer() = Timer(NaN, NaN)

timer_start(self::Timer) = self.start_time = time()

timer_stop(self::Timer) = self.stop_time = time()

call(self::Timer) = isnan(self.stop_time) ? time() - self.start_time : self.stop_time - self.start_time

Base.show(io::IO, self::Timer) = print(io, "time => ", round(self(), 2))

type Progress{C<:Component,E<:AbstractEvaluation,S<:AbstractStoppingCriteria}
    model::C
    best_model::C
    evaluate::E
    criteria::S
    timer::Timer
    min_epochs::Int
    max_epochs::Real
    frequency::Int
    epoch::Int
    epoch_pad::Int
    converged::Bool
end

function Base.show(io::IO, self::Progress)
    epoch = lpad(self.epoch, self.epoch_pad, " ")
    print(io, "[$epoch] $(self.criteria), $(self.timer), $(self.evaluate)")
end


function Progress(
    model::Component, 
    evaluate::AbstractEvaluation, 
    criteria::AbstractStoppingCriteria;
    min_epochs::Int=2,
    max_epochs::Real=20,
    frequency::Int=1,
    start::Bool=true
    )
    timer = Timer()
    epoch_pad = max_epochs < Inf ? ceil(Int, log10(max_epochs + 1)) : 0
    self = Progress(model, model, evaluate, criteria, timer, min_epochs, max_epochs, frequency, 0, epoch_pad, false)
    start && timer_start(self.timer) 
    return self
end

converged(self::Progress) = self.converged

timer_start(self::Progress) = timer_start(self.timer)

timer_stop(self::Progress) = timer_stop(self.timer)

Base.time(self::Progress) = self.timer()

evaluate(self::Progress; best::Bool=false) = best ? self.evaluate.best : self.evaluate.curr

epoch(self::Progress) = self.epoch

best(self::Progress) = self.best_model

function update_progress!(self::Progress, improved::Bool, value::Real, save::Bool)
    converged = self.criteria(improved, value)
    if self.epoch < self.min_epochs
        self.converged = false
    elseif self.epoch >= self.max_epochs
        self.converged = true
    else
        self.converged = converged
    end

    if save && improved
        self.best_model = deepcopy(self.model)
    end

    return self
end

function call{C,E<:AbstractExternalEvaluation,S}(self::Progress{C,E,S}, curr::Number; save::Bool=false)
    self.epoch += 1
    if self.epoch % self.frequency == 0
        improved = self.evaluate(curr)
        update_progress!(self, improved, self.evaluate.curr, save)
        return true
    else
        return false
    end
end

function call{C,E<:AbstractFunctionEvaluation,S}(self::Progress{C,E,S}; save::Bool=false)
    self.epoch += 1
    if self.epoch % self.frequency == 0
        improved = self.evaluate()
        update_progress!(self, improved, self.evaluate.curr, save)
        return true
    else
        return false
    end
end
