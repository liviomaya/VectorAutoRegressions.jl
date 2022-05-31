module VectorAutoRegressions

export VAR, setvar, size, isstable, irf, mean, mode, cov, cor, vardecomp,
    NormalInvWishart, rand, setniw, fitols, bayesreg, VARNormalInvWishart,
    MinnesotaPrior, setmnprior, bvar, BVARSample, map, hpbvar, shrinkplot,
    MHOptions, kalmanfilter, kalmansmoother, emfill, forecast, cholesky


include(pwd() * "/src/f0_header.jl")
include(pwd() * "/src/f1_struct.jl")
include(pwd() * "/src/f2_varfunc.jl")
include(pwd() * "/src/f3_niw.jl")
include(pwd() * "/src/f4_estimate.jl")
include(pwd() * "/src/f5_hpestimate.jl")

end # module
