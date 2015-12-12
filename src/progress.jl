type Progress{T<:Component}
    # parameters
    f::Nullable{Function}
    model::T
    tol::Number
    minimize::Bool
    min_epochs::Int
    max_epochs::Int
    patience::Int
    precision::Int
    # state
    epoch_pad::Int
    frustration_pad::Int
    start_time::Float64
    stop_time::Float64
    best_model::T
    best_value::Number
    current_value::Number
    epoch::Int
    frustration::Number
    should_stop::Bool
    improved::Bool
end

function Progress(
    f::Function,
    model::Component;
    tol::Number=1e-3,
    minimize::Bool=true,
    min_epochs::Int=2,
    max_epochs::Int=20,
    patience::Int=1,
    precision::Int=2,
    )
    return Progress(
        # parameters
        Nullable{Function}(f),
        model,
        tol,
        minimize,
        min_epochs,
        max_epochs,
        patience,
        precision,
        # state
        ceil(Int, log10(max_epochs + 1)),
        ceil(Int, log10(patience + 1)),
        0.0,
        0.0,
        model,
        minimize ? Inf : -Inf,
        minimize ? Inf : -Inf,
        0,
        0,
        false,
        false
    )
end

function Progress(
    model::Component;
    tol::Number=1e-3,
    minimize::Bool=true,
    min_epochs::Int=2,
    max_epochs::Int=20,
    patience::Int=1,
    precision::Int=3,
    )
    return Progress(
        # parameters
        Nullable{Function}(),
        model,
        tol,
        minimize,
        min_epochs,
        max_epochs,
        patience,
        precision,
        # state
        ceil(Int, log10(max_epochs + 1)),
        ceil(Int, log10(patience + 1)),
        0.0,
        0.0,
        model,
        minimize ? Inf : -Inf,
        minimize ? Inf : -Inf,
        0,
        0,
        false,
        false
    )
end

function Base.show(io::IO, self::Progress)
    epoch = lpad(self.epoch, self.epoch_pad, " ")
    frustration = lpad(self.frustration, self.frustration_pad, " ")
    current = round(self.current_value, self.precision)
    best = round(self.best_value, self.precision)
    print(io, "epoch: $epoch, frustration: $frustration, current: $current, best: $best")
end

function Base.step(self::Progress, current_value::Number; store_best::Bool=false)
    self.epoch += 1

    self.current_value = current_value
    if self.minimize
        self.improved = self.best_value - self.current_value > self.tol
        self.best_value = min(self.best_value, self.current_value)
    else
        self.improved = self.best_value - self.current_value < self.tol
        self.best_value = max(self.best_value, self.current_value)
    end

    if self.improved
        self.frustration = 0
    else
        self.frustration += 1
    end

    if store_best && self.improved
        self.best_model = deepcopy(self.model)
    end

    self.should_stop = if self.epoch >= self.max_epochs
        true
    elseif self.epoch <= self.min_epochs
        false
    else
        self.frustration > self.patience
    end
    return nothing
end

function Base.step(self::Progress; store_best::Bool=false)
     isnull(self.f) && error("must provide current value if no evaluation function provided")
     step(self, get(self.f)(), store_best=store_best)
     return nothing
 end

Base.start(self::Progress) = self.start_time = time()
Base.done(self::Progress) = self.stop_time = time()
Base.quit(self::Progress) = self.should_stop
Base.time(self::Progress) = self.stop_time - self.start_time
