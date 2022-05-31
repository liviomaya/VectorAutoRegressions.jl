function converthp(hyppar::Vector{<:Real}, unitroot::Vector{Bool})
    N = length(unitroot)
    μ = zeros(N)
    ψ = hyppar[1:N]
    D = hyppar[N+1]
    λ = hyppar[N+2]
    γ = hyppar[N+3]
    δ = hyppar[N+4]
    prior = setmnprior(μ, unitroot, ψ, D + N + 1, λ, γ, δ)
    return prior
end


"""
    sample, marginal, priormode = hpbvar(data, hpprior; <kwargs>)

Sample from the posterior of hyperparameters. Uses the Metropolis-Hastings algorithm.

### Arguments

- `data::VecOrMat{<:Real}`: array with sample (observations in rows, variables in columns).

- `hpprior`: vector of `Distribution` objects (see `Dsistributions` package) containing the prior of the hyperparameters. The first `N` entries should have the distributions of the modes of the Inverse-Wishart distributions that form the prior to `Σ` (where `N` is the number of variables in the data, and `Σ` the covariance matrix of residuals).  Entry `N+1` gives the distribution for (adjusted) degrees of these IW distributions. The next three entries give distributions to the shrinkage parameters (unit root prior, sum of coefficients and co-persistence). 
 
### Output

- `sample::BVARSample`: sample of priors and main parameters (in the form of `VAR` objects). 

- `marginal::Float64`: is the marginal likelihood of the model.

- `priormode::MinnesotaPrior`: prior build from the mode of the hyperparameters' distribution.

### Keyword Arguments

- `unitroot::Vector{Bool}=trues(N)`: select which variables are unit roots in the prior; others are white noise.

- `P::Int64`: number of lags in the estimated VAR.

- `intercept::Bool=true`: whether the VAR model to be fitted contains a constant term.

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `data`  correspond to the same period.

- `options::MHOptions`: options for the Metropolis-Hastings algorithm (see `MetropolisHastings` package).
"""
function hpbvar(data::VecOrMat{<:Real}, hpprior;
    unitroot::Vector{Bool}=Vector(trues(size(data, 2))),
    P::Int64=1,
    intercept::Bool=true,
    fp::VecOrMat{U}=zeros(size(data)),
    options::MHOptions=MHOptions()) where {U<:Real}

    # set up prior function
    function prior(hp::Vector{<:Real})
        any(hp .< minimum.(hpprior)) && return -Inf
        any(hp .> maximum.(hpprior)) && return -Inf
        return Distributions.logpdf.(hpprior, hp) |> sum
    end

    # set up likelihood
    function likelihood(hp::Vector{<:Real})
        mnprior = converthp(hp, unitroot)
        marginal = bvar(data, mnprior, P=P, intercept=intercept, fp=fp)[1]
        isnan(marginal) && (marginal = -Inf)
        return marginal
    end

    x0 = median.(hpprior) # usemedian function as mode is not defined for truncated distributions
    results = mhsampler(x0, prior, likelihood, options=options)
    marg = results.marginal

    # build BVARSample
    T = size(results.sample)[1]
    priorsamp = Vector{MinnesotaPrior}(undef, T)
    varsamp = Vector{VAR}(undef, T)
    progress = Progress(T, dt=1, barlen=25)
    for t in 1:T
        hp = results.sample[t, :]
        mnprior = converthp(hp, unitroot)
        priorsamp[t] = mnprior
        posterior = bvar(data, mnprior, P=P, intercept=intercept, fp=fp)[2]
        varsamp[t] = rand(posterior)
        next!(progress)
    end
    sample = BVARSample(priorsamp, varsamp)

    # build prior mode
    hpmode = results.mode
    priormode = converthp(hpmode, unitroot)

    return sample, marg, priormode
end

"""
    shrinkplot(hpprior, sample::BVARSample)

Plot the prior and posterior distributions of the shrinkage parameters `λ`, `γ` and `δ` (referring, respectivelly, to the unit root, sum-of-coefficients and co-persistence priors).
"""
function shrinkplot(hpprior, sample::BVARSample)
    T = length(sample.prior)
    P = length(hpprior)
    figarray = Vector{Plots.Plot}(undef, 0)
    for (field, hpindex) in zip([:λ, :γ, :δ], [P - 2, P - 1, P])
        fieldsample = [getfield(sample.prior[t], field) for t in 1:T]
        dist = hpprior[hpindex]
        if typeof(dist) == Truncated{Gamma{Float64},Continuous,Float64}
            untruncdist = dist.untruncated
        else
            untruncdist = dist
        end
        fitdist = fit(typeof(untruncdist), fieldsample)

        fig = histogram(fieldsample,
            title=string(field),
            label="Sample",
            legend=(field == :λ) ? :best : :none,
            alpha=0.30,
            bins=75,
            color=2,
            lw=0.1,
            margin=4mm,
            normalize=:true,
            # size=120 * [3, 3],
            tickfontsize=12,
            titlefontsize=14,
            legendfontsize=10,
            color_palette=:seaborn_bright,
            gridalpha=0.2,
            gridlinesstyle=:dot,
            fg_legend=:transparent)

        plot!(fig, x -> pdf(fitdist, x),
            lw=1,
            label="Sample Fit",
            color=:black)

        plot!(fig, x -> pdf(dist, x),
            lw=1,
            label="Prior",
            color=:black,
            ls=:dash)

        push!(figarray, fig)
    end
    fig = plot(figarray..., layout=(1, 3), size=140 * [9, 3])

    display(fig)
    return
end


