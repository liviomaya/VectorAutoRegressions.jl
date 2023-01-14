# -----------------------------------------------------------------------
# USEFUL FUNCTIONS FOR VAR OBJECTS
# -----------------------------------------------------------------------

function isstable(V::StackedVAR; report::Bool=false)
    vals = eigen(V.Φ).values
    absvals::Vector{Float64} = abs.(vals)
    dom = maximum(absvals)
    stable = dom < 1
    !report && return stable
    df = DataFrame(:EigenValues => vals, :Abs_Val => absvals)
    sort!(df, :Abs_Val, rev=true)
    digits = (v, i, j) -> round(v, digits=2)
    pretty_table(df, title="System Eigenvalues", formatters=digits)
    message = (stable) ? "Stationary System." : "Non-Stationary System!"
    println(message)
    return stable
end

"""
    isstable(v::VAR; report::Bool=true)

Return `true` if the largest eigenvalue of the `VAR` `v` is less than one. If
`report` is `true`, show table with eigenvalues.
"""
isstable(v::VAR; report::Bool=false) = isstable(stack(v), report=report)

function ma(V::StackedVAR, T::Int64)
    N, R, Q = size(V)
    K = zeros(R, Q, T)
    for t in 1:T
        K[:, :, t] = (V.Φ^(t - 1)) * V.Γ
    end
    Ktrim = K[1:N, :, :]
    return Ktrim
end
ma(v::VAR, T::Int64) = ma(stack(v), T)

function mean(v::StackedVAR)
    @assert isstable(v) "System not covariance-stationary."
    R = size(v, 2)
    ltmean = (I(R) .- v.Φ) \ v.μ
    return ltmean
end

"""
    mean(v::VAR)

Compute the unconditional mean of the variables in the VAR `v`. 
"""
mean(v::VAR) = mean(stack(v))[1:size(v, 1)]


function cov(v::StackedVAR)
    @assert isstable(v) "System not covariance-stationary."
    R = size(v, 2)
    covvec = (I(R^2) .- kron(v.Φ, v.Φ)) \ (v.Γ*v.Σ*v.Γ')[:]
    V = reshape(covvec, R, R)
    V = round.(V, digits=10) |> makehermitian
    return V
end

"""
    cov(v::VAR)

Compute the covariance matrix of the variables in the VAR `v`. 
"""
cov(v::VAR) = cov(stack(v))[1:size(v, 1), 1:size(v, 1)]

"""
    cor(v::VAR)

Compute the correlation matrix of the variables in the VAR `v`. 
"""
function cor(v::T) where {T<:AbstractVAR}
    V = cov(v)
    stdev = sqrt.(diag(V))
    stmat = diagm(0 => stdev)
    istmat = pinv(stmat) # pseudo-inverse
    R = istmat * V * istmat
    return R
end

function condcov(v::StackedVAR, T::Int64)
    R = size(v, 2)
    V = zeros(R, R, T)
    for t in 1:T
        lastmat = (t == 1) ? zeros(R, R) : V[:, :, t-1]
        V[:, :, t] = v.Φ * lastmat * v.Φ' + v.Γ * v.Σ * v.Γ'
    end
    return V
end
condcov(v::VAR, T::Int64) = condcov(stack(v), T)[1:size(v, 1), 1:size(v, 1), :]


function irfplot(path, lowerbo, upperbo, options)

    T, N = size(path)

    fig = plot(1:T, path;
        xlim=(1, Inf),
        size=120 * [3, 3],
        fillalpha=0.2,
        markershape=:circle,
        msw=0,
        color_palette=:seaborn_bright,
        options...
    )

    plot!(fig, lowerbo, fillrange=upperbo,
        label=:none,
        lw=0,
        fillalpha=0.2,
        color=permutedims(1:N))

    hline!(fig, [0.0], label=:none, color=:black, alpha=0.15)

    return fig
end

"""

    path, lb, ub = irf(v::VAR, shock::Vector{<:Real}; <kwargs>)
    path, lb, ub = irf(v::VAR, shock::Int64; <kwargs>)

Return the response (i.e. change in expectation) of variables in the VAR `v` to impulse `shock`.

### Arguments

- `v::VAR`: `VAR` model to compute impulse response function. 
    
- `shock::Union{Vector, Int64}`: impulse to the VAR. If `shock` is an `Int64`, `irf` computes the response to a impulse of one unit in corresponding variable. 

### Output

- `path`: array with the response. Time in rows, variables in columns.
    
- `lb`: array with lower limit of the confidence interval.

- `ub`: array with upper limit of the confidence interval.

### Keyword Arguments

- `T::Int64=10`: length of computed response.
    
- `id::AbstractVector=1:size(v,1)`: index of variables to which responses are calculated.
    
- `cover::Float64=0`: probability coverage of the confidence interval.

- `plot::Bool=true`: whether to plot the IRF.

- `options::Dict`: options to figure (see documentation to `Plots` package). 
"""
function irf(v::H, shock::Vector{U};
    T::Int64=10,
    id::AbstractVector{J}=1:size(v, 1),
    cover::Float64=0.0,
    plot::Bool=true,
    options::Dict{D1,D2}=Dict(:label => id')) where {H<:AbstractVAR,U<:Real,J<:Real,D1,D2}

    maar = ma(v, T)
    maar = mapslices(m -> m * shock, maar, dims=(1, 2))
    maar = round.(maar, digits=12)
    irfdata::Matrix{Float64} = maar[id, 1, :]' |> collect

    # compute ribbon
    ribbon = zeros(T, length(id))
    if cover > 0
        covar = condcov(v, T - 1)
        N = size(covar, 1)
        varar = mapslices(diag, covar, dims=(1, 2))
        varar = reshape(varar, N, T - 1) |> permutedims
        varar = [zeros(1, N); varar]
        stdar = sqrt.(varar)
        ribbon = stdar[:, id] * cquantile(Normal(), (1 - cover) / 2)
    end

    lowerbo::Matrix{Float64} = irfdata .- ribbon
    upperbo::Matrix{Float64} = irfdata .+ ribbon
    plot && irfplot(irfdata, lowerbo, upperbo, options) |> display

    return irfdata, lowerbo, upperbo
end

irf(v::H, shock::Int64;
    T::Int64=10,
    id::AbstractVector{J}=1:size(v, 1),
    cover::Float64=0.0,
    plot::Bool=true,
    options::Dict{D1,D2}=Dict()) where {H<:AbstractVAR,U<:Real,J<:Real,D1,D2} =
    irf(v, collect(I(size(v, 3)))[:, shock], T=T, id=id,
        cover=cover, plot=plot, options=options)


function vardecomp(v::VAR, T::Int64)
    N, P, Q = size(v)
    mav = ma(v, T)
    V = zeros(N, Q)
    for q in 1:Q
        Iq = zeros(Q, Q)
        Iq[q, q] = 1
        V[:, q] = diag(sum(mav[:, :, t] * Iq * v.Σ * mav[:, :, t]'
                           for t in 1:T))
    end
    return V
end

"""
    V, R = cholesky(v::VAR)

Orthogonalize innovations of VAR `v` and return new VAR `V`.

    `v` model: xₜ = μ + Ψ xₜ₋₁ + Γ ϵₜ        cov(ϵ) = Σ
    
    `V` model: xₜ = μ + Ψ xₜ₋₁ + ΓR ηₜ       cov(η) = I
"""
function cholesky(v::VAR)
    # ηₜ = R⁻¹ ϵₜ       cov(η) = I
    G = Matrix(cholesky(v.Σ).L) # Σ = G G'
    # I = cov(η) = R⁻¹ Σ R = R⁻¹ G G' R⁻¹' => R = G
    R = G
    # MA: xₜ = ∑ Cᵢ ϵₜ₋₁ = ∑ Cᵢ R R⁻¹ ϵₜ₋₁ = ∑ Cᵢ R ηₜ₋ᵢ
    # State Space: xₜ = P xₜ₋₁ + Q ϵₜ = P xₜ₋₁ + (Q R) ηₜ
    newΓ = v.Γ * R
    V = setvar(v.μ, v.Ψ, newΓ, I(size(v, 3)))
    return V, R
end

function rand(v::StackedVAR, T::Int64, burn::Int64, fp::VecOrMat{<:Real})

    N, R, Q = size(v)
    shockdist = MvNormal(v.Σ)

    # burn sample
    s0 = isstable(v) ? mean(v) : zeros(R)
    for _ in 1:burn
        shock = rand(shockdist)
        s0 = v.μ + v.Φ * s0 + v.Γ * shock
    end

    # simulation
    path = zeros(T, R)
    for t in 1:T
        state = (t == 1) ? s0 : path[t-1, :]
        shock = rand(shockdist)
        path[t, :] = v.μ + v.Φ * state + fp[t, :] + v.Γ * shock
    end

    return path
end

"""
    S = rand(v::VAR[, T]; <kwargs>)

Draw a random sample of VAR model `v`. The initial value is the unconditional mean of the VAR if it is stable and zero if not.

Output `S` contains draws in rows and variables in columns.

### Arguments

- `v::VAR`: `VAR` object with model to draw from.

- `T::Int64=1`: sample size.

### Keyword Arguments

- `burn::Int64`: number of burn in draws.

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and output `S` correspond to the same period. `fp` assumed to be zero in the burn-in sample.

"""
rand(v::VAR, T::Int64; burn::Int64=0,
    fp::VecOrMat{<:Real}=zeros(T, size(v, 1))) =
    rand(stack(v), T, burn, stackfp(fp, v))[:, 1:size(v, 1)]


function stack(data::Matrix{U}, P::Int64=1) where {U<:Real}
    T, N = size(data)
    @assert T >= P
    extdata = [NaN * ones(P - 1, N); data]
    stackdata = mapreduce(permutedims ∘ vec ∘ permutedims, vcat,
        [extdata[t.+(P:-1:1), :] for t in 0:T-1])
    return stackdata
end
stack(data::Vector{U}, P::Int64=1) where {U<:Real} = stack(mat(data), P)

function forecast(states::VecOrMat{<:Real}, V::StackedVAR,
    periods::Int64, fp::VecOrMat{U}) where {U<:Real}
    T = size(states, 1)
    fcast = copy(states)
    for i in 1:periods
        fcast = ones(T) * V.μ' + fcast * V.Φ' + fp[i.+(1:T), :]
    end
    return fcast
end

"""
    F = forecast(states, v, horizon; fp)

Forecast the path of variables in VAR `v`.

### Arguments

- `states::VecOrMat{<:Real}`: array with states in each row.

- `v::VAR`: `VAR` object with model used in the forecast.

- `horizon::Int64`: horizon of the forecast. If `horizon=0`, return `states`.

### Output

- `F::Matrix{Float64}`: stores the forecast for each state in `states` in the corresponding row. The first `P` forecasts are `NaN` by default (where `P` is the number of non-zero lags in VAR `v`.) 

### Keyword Argument

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `states`  correspond to the same period. Therefore `fp` must have at least `horizon` rows more than `states`. 
"""
function forecast(states::VecOrMat{<:Real}, v::VAR, horizon::Int64;
    fp::VecOrMat{U}=
    zeros(size(states, 1) + horizon, size(v)[1])) where {U<:Real}
    N, P = size(v)[1:2]
    fcast = forecast(stack(states, P), stack(v), horizon, stackfp(fp, v))
    fcast = fcast[:, 1:N]
    return fcast
end

"""
    F = forecast(states, v, horizon::Vector{Int64}; fp)

Forecast the path of variables in VAR `v` for all horizons in `horizon`. Return a vector of matrices with each one corresponding to a different horizon.
"""
forecast(states::VecOrMat{<:Real}, v::VAR, horizon::Vector{Int64};
    fp::VecOrMat{U}=
    zeros(size(states, 1) + maximum(horizon), size(v, 1))) where {U<:Real} =
    [forecast(states, v, t, fp=fp) for t in horizon]

function fcastextend(data::VecOrMat{<:Real}, V::StackedVAR,
    periods::Int64, fp::VecOrMat{<:Real})

    data_last = permutedims(data[end, :])
    fp_last = fp[end-periods:end, :]
    fcast = copy(data)
    for p in 1:periods
        fcast = [fcast; forecast(data_last, V, p, fp_last)]
    end

    return fcast
end

"""
    F = fcastextend(states, v, horizon; fp)

Extend `states` array with forecasts under model `v`.

### Arguments

- `states::VecOrMat{<:Real}`: array with states in each row.

- `v::VAR`: `VAR` object with model used in the forecast.

- `horizon::Int64`: horizon of the forecast. If `horizon=0`, return `states`.

### Output

- `F::Matrix{Float64}`: extended array

### Keyword Argument

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `states`  correspond to the same period. Therefore `fp` must have at least `horizon` rows more than `states`. 
"""
function fcastextend(data::VecOrMat{<:Real}, v::VAR,
    periods::Int64; fp::VecOrMat{U}=zeros(size(data, 1) + periods, size(v, 1))) where {U<:Real}
    N, P = size(v)[1:2]
    fcast = fcastextend(stack(data, P), stack(v), periods,
        stackfp(fp, v))
    return fcast[:, 1:P]
end


#--------------------------------------------------------------------
# KALMAN FILTER AND SMOOTHER
#--------------------------------------------------------------------

function kaldefaultinitial(v::VAR)::Tuple{Vector{Float64},Matrix{Float64}}
    V = stack(v)
    if isstable(V)
        initialmean = mean(V)
        initialcov = cov(V)
    else
        P = size(v, 2)
        R = size(V, 2)
        # these array are irrelevant if all variables are observed in the first P periods
        initialmean = zeros(R)
        initialcov = kron(collect(I(P)), v.Σ)
    end
    return initialmean, initialcov
end

function kalmanfilter(X::VecOrMat{<:Real},
    V::StackedVAR,
    fp::VecOrMat{<:Real},
    μ0::Vector{<:Real},
    Σ0::Matrix{<:Real})

    N, R = size(V)[[1, 2]]
    P = Int64(R / N)
    @assert size(X, 2) == N

    # s = E(X(t) | t-1 information)
    # S = E(X(t) | t information)
    # q = Cov(X(t) | t-1 information)
    # Q = Cov(X(t) | t information)

    T = size(X, 1)
    s::Matrix{Float64} = zeros(T, R)
    S::Matrix{Float64} = zeros(T, R)
    q::Array{Float64,3} = zeros(T, R, R)
    Q::Array{Float64,3} = zeros(T, R, R)
    likelihood::Float64 = 0.0
    K::Matrix{Float64} = zeros(R, N)

    # unpack parameters
    μ = V.μ
    Φ = V.Φ
    Γ = V.Γ
    Σ = V.Σ

    # relevant objects
    # myround(x::Float64) = round(x, digits=20)
    eye = collect(I(R))
    Λ = eye[1:N, :] # select observables
    ΓΣΓ = Γ * Σ * Γ'
    withfp = any(fp .!= 0)

    # filter
    for t in 1:T
        Slast = (t == 1) ? μ0 : S[t-1, :]
        Qlast = (t == 1) ? Σ0 : Q[t-1, :, :]

        # project t based on t-1 information
        s[t, :] = μ .+ Φ * Slast
        q[t, :, :] = makehermitian(Φ * Qlast * Φ' + ΓΣΓ)
        withfp && (s[t, :] .+= fp[t, :])

        # define variables observed in period t
        isobs = .!isnan.(X[t, :])
        if count(isobs) == 0
            S[t, :] = s[t, :]
            Q[t, :, :] = q[t, :, :]
            continue
        end
        yproj = Λ[isobs, :] * s[t, :] # projected to be observed based on t-1 information
        y = X[t, isobs] # observed
        F = Λ[isobs, :] * q[t, :, :] * Λ[isobs, :]' # |> makehermitian # t-1 conditional covariance of observables

        # project t based on t information
        invF = inv(F)
        K = q[t, :, :] * Λ[isobs, :]' * invF
        S[t, :] = s[t, :] + K * (y .- yproj)
        auxm = eye - K * Λ[isobs, :]
        Q[t, :, :] = (auxm * q[t, :, :] * auxm') |> makehermitian

        # marginal
        likelihood -= (1 / 2) * count(isobs) * log(2 * pi)
        likelihood -= (1 / 2) * log(det(F))
        likelihood -= (1 / 2) * (y - yproj)' * invF * (y - yproj)
    end

    return S, Q, s, q, likelihood, K
end

"""
    states, covstates, likelihood = kalmanfilter(X, v; <kwargs>)

Run the Kalman filter.

### Arguments

- `X::VecOrMat{<:Real}`: array with data points in each row. Missing observations should be entered as `NaN`. It is not necessary to stack `X` in any way.

- `v::VAR`: `VAR` object with model used in to filter.

### Output

- `states::Matrix{Float64}`: Expected value of the distribution of states conditional on current information (`states[t,:]` contains expectation conditional on observation of variables up to period `t`).

- `covstates::Array{Float64, 3}`: Covariance matrices of the distributions conditional on current information (`covstates[t,:,:]` contains expectation conditional on observation of variables up to period `t`).

- `likelihood::Float64`: Log-likelihood of `X`.

### Keyword Arguments

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `X`  correspond to the same period. 

- `μ0::Vector{<:Real}=mean(v)`: mean of the probability distribution of the initial condition. If `v` is not stable, default to zero (if the first `P` observations do not have missing values, initial condition does not matter).

- `Σ0::Matrix{<:Real}=cov(v)`: covariance matrix of the probability distribution of the initial condition. If `v` is not stable, default to identity (if the first `P` observations do not have missing values, initial condition does not matter).

"""
function kalmanfilter(X::VecOrMat{<:Real}, v::VAR;
    fp::VecOrMat{T}=zeros(size(X)),
    μ0::Vector{<:Real}=kaldefaultinitial(v)[1],
    Σ0::Matrix{<:Real}=kaldefaultinitial(v)[2]) where {T<:Real}

    N = size(v, 1)
    S, Q, likelihood::Float64 = kalmanfilter(X, stack(v), stackfp(fp, v),
        μ0, Σ0)[[1, 2, 5]]
    S::Matrix{Float64} = S[:, 1:N]
    Q::Array{Float64,3} = Q[:, 1:N, 1:N]
    return S, Q, likelihood
end

function kalmansmoother(X::VecOrMat{<:Real},
    V::StackedVAR,
    fp::VecOrMat{<:Real},
    μ0::Vector{<:Real},
    Σ0::Matrix{<:Real};
    getWlag::Bool=false)

    # getVlag relevant for EM algorithm

    S, Q, s, q, lhd, K = kalmanfilter(X, V, fp, μ0, Σ0)

    N, R = size(V)

    # M = E(X(t) | 1:T information)
    # W = cov(X(t) | 1:T information)

    T = size(X, 1)
    Z = zeros(T, R)
    W = zeros(T, R, R)
    getWlag && (J = zeros(T, R, R))

    # unpack parameters
    Φ = V.Φ
    eye = collect(I(R))
    Λ = eye[1:N, :] # select observables

    # filter 
    Z[T, :] = S[T, :]
    W[T, :, :] = Q[T, :, :]
    for t in T-1:-1:1
        Jt = Q[t, :, :] * Φ' * pinv(q[t+1, :, :])
        getWlag && (J[t, :, :] .= Jt)
        Z[t, :] = S[t, :] .+ Jt * (Z[t+1, :] .- s[t+1, :])
        W[t, :, :] = Q[t, :, :] .+ Jt * (W[t+1, :, :] .- q[t+1, :, :]) * Jt'
    end

    # period-0 distribution
    J0 = Σ0 * Φ' * pinv(q[1, :, :])
    Z0::Vector{Float64} = μ0 .+ J0 * (Z[1, :] .- s[1, :])
    W0::Matrix{Float64} = Σ0 .+ J0 * (Q[1, :, :] .- q[1, :, :]) * J0'

    # compute Vlag[t,:,:] = cov(X(t), X(t-1) | 1:T information)
    Wlag::Array{Float64,3} = zeros(T, R, R)
    if getWlag
        isobs = .!isnan.(X[T, :])
        Wlag[T, :, :] = (I(R) .- K * Λ[isobs, :]) * Φ * Q[T-1, :, :]
        for t in T-1:-1:1
            Jt = (t == 1) ? J0 : J[t-1, :, :]
            Wlag[t, :, :] .= Q[t, :, :] * Jt' +
                             J[t, :, :] * (Wlag[t+1, :, :] - Φ * Q[t, :, :]) *
                             Jt'

        end
    end

    return Z, W, Z0, W0, Wlag, lhd
end

"""
    states = kalmansmoother(X, v; <kwargs>)

Run the Kalman smoother.

### Arguments

- `X::VecOrMat{<:Real}`: array with data points in each row. Missing observations should be entered as `NaN`. It is not necessary to stack `X` in any way.

- `v::VAR`: `VAR` object with model used in to filter.

### Output

- `states::Matrix{Float64}`: Expected value of the distribution of states conditional on current information (`states[t,:]` contains expectation conditional on observation of variables up to period `t`).

### Keyword Arguments

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `X`  correspond to the same period. 

- `μ0::Vector{<:Real}=mean(v)`: mean of the probability distribution of the initial condition. If `v` is not stable, default to zero (if the first `P` observations do not have missing values, initial condition does not matter).

- `Σ0::Matrix{<:Real}=cov(v)`: covariance matrix of the probability distribution of the initial condition. If `v` is not stable, default to identity (if the first `P` observations do not have missing values, initial condition does not matter).

"""
function kalmansmoother(X::VecOrMat{<:Real}, v::VAR;
    fp::VecOrMat{U}=zeros(size(X)),
    μ0::Vector{G}=kaldefaultinitial(v)[1],
    Σ0::Matrix{U}=kaldefaultinitial(v)[2]) where {G<:Real,U<:Real}

    T = size(X, 1)
    N, P = size(v)[[1, 2]]

    if !any(isnan.(X))
        return X, zeros(T, N, N)
    end

    # avoid costly computation of deterministic first part of the data
    inan = .!prod(.!isnan.(X), dims=2)[:] |> findfirst # first row with NaN
    r1::Matrix{Float64} = zeros(T, N)
    r2::Array{Float64,3} = zeros(T, N, N)
    if inan <= P
        M, W = kalmansmoother(X, stack(v), stackfp(fp, v), μ0, Σ0)[[1, 2]]
        r1 = M[:, 1:N]
        r2 = W[:, 1:N, 1:N]
    else
        M, W = kalmansmoother(X[inan-P:end, :], stack(v),
            stackfp(fp[inan-P:end, :], v), μ0, Σ0)[[1, 2]]

        r1 = [X[1:inan-P-1, :]; M[:, 1:N]]
        r2 = zeros(T, N, N)
        r2[inan-P:end, :, :] .= W[:, 1:N, 1:N]
    end
    return r1, r2
end
