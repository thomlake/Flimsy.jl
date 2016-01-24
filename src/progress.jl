
abstract AbstractCriterion

type Criterion <: AbstractCriterion
    minimize::Bool
    tol::Number
    precision::Int
    curr::Number
    best::Number
    improved::Bool
end

function Criterion(; minimize::Bool=true, tol::Number=1e-3, precision::Int=3)
    initial = minimize ? Inf : -Inf
    return Criterion(minimize, tol, precision, initial, initial, false)
end

function call(self::Criterion, curr::Number)
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

type FunctionCriterion <: AbstractCriterion
    f::Function
    minimize::Bool
    tol::Number
    precision::Int
    curr::Number
    best::Number
    improved::Bool
end

function FunctionCriterion(f::Function; minimize::Bool=true, tol::Number=1e-3, precision::Int=3)
    initial = minimize ? Inf : -Inf
    return FunctionCriterion(f, minimize, tol, precision, initial, initial, false)
end

function call(self::FunctionCriterion)
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

function Base.show(io::IO, self::AbstractCriterion)
    curr = round(self.curr, self.precision)
    best = round(self.best, self.precision)
    print(io, "curr => $curr, best => $best")
end


type Patience
    patience::Int
    frustration::Int
    frustration_pad::Int
end

Patience(patience, frustration) = Patience(patience, frustration, ceil(Int, log10(patience + 1)))

Patience(patience::Int=1) = Patience(patience, 0)

function call(self::Patience, improved::Bool)
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
    print(io, "frustration => ($frustration of $patience)")
end

type Timer
    start_time::Float64
    stop_time::Float64
end

Timer() = Timer(NaN, NaN)

Base.start(self::Timer) = self.start_time = time()

stop(self::Timer) = self.stop_time = time()

call(self::Timer) = isnan(self.stop_time) ? time() - self.start_time : self.stop_time - self.start_time

Base.show(io::IO, self::Timer) = print(io, "time => ", round(self(), 2))

type Progress{M<:Component,C<:AbstractCriterion}
    model::M
    best_model::M
    criteria::C
    patience::Patience
    timer::Timer
    min_epochs::Int
    max_epochs::Int
    epoch::Int
    epoch_pad::Int
    converged::Bool
end

function Base.show(io::IO, self::Progress)
    epoch = lpad(self.epoch, self.epoch_pad, " ")
    print(io, "[$epoch] $(self.patience), $(self.timer), $(self.criteria)")
end

function Progress(model::Component, criteria::AbstractCriterion; min_epochs::Int=2, max_epochs::Int=20, kwargs...)
    kwdict = Dict(kwargs)

    patience = haskey(kwdict, :patience) ? Patience(kwdict[:patience]) : Patience()

    timer = Timer()
    epoch_pad = max_epochs < Inf ? ceil(Int, log10(max_epochs + 1)) : 0
    self = Progress(model, model, criteria, patience, timer, min_epochs, max_epochs, 0, epoch_pad, false)

    if haskey(kwdict, :start)
        start(self.timer)
    end

    return self
end

function Progress(model::Component; min_epochs::Int=2, max_epochs::Int=20, kwargs...)
    kwdict = Dict(kwargs)
    dcriteria = Dict()
    for kw in [:minimize, :tol, :precision]
        if haskey(kwdict, kw)
            dcriteria[kw] = pop!(kwdict, kw)
        end
    end
    criteria = Criterion(dcriteria...)
    return Progress(model, criteria; min_epochs=min_epochs, max_epochs=max_epochs, kwdict...)
end

function Progress(f::Function, model::Component; min_epochs::Int=2, max_epochs::Int=20, kwargs...)
    kwdict = Dict(kwargs)
    dcriteria = Dict()
    for kw in [:minimize, :tol, :precision]
        if haskey(kwdict, kw)
            dcriteria[kw] = kwdict[kw]
        end
    end
    criteria = FunctionCriterion(f; dcriteria...)
    return Progress(model, criteria; min_epochs=min_epochs, max_epochs=max_epochs, kwdict...)
end

converged(self::Progress) = self.converged

Base.start(self::Progress) = start(self.timer)

stop(self::Progress) = stop(self.timer)

Base.time(self::Progress) = self.timer()

criteria(self::Progress, best=true) = best ? self.criteria.best : self.criteria.curr

epoch(self::Progress) = self.epoch

best(self::Progress) = self.best_model

function update_state(self::Progress, improved::Bool, frustrated::Bool, save::Bool)
    self.epoch += 1
    if self.epoch < self.min_epochs
        self.converged = false
    elseif frustrated || self.epoch >= self.max_epochs
        self.converged = true
    else
        self.converged = false
    end

    if save && improved
        self.best_model = deepcopy(self.model)
    end

    return self
end

function call{M}(self::Progress{M,Criterion}, curr::Number; save::Bool=false)
    improved = self.criteria(curr)
    frustrated = self.patience(improved)
    return update_state(self, improved, frustrated, save)
end

function call{M}(self::Progress{M,FunctionCriterion}; save::Bool=false)
    improved = self.criteria()
    frustrated = self.patience(improved)
    return update_state(self, improved, frustrated, save)
end
