using ColorSchemes
using DataFrames, Distributions
using LinearAlgebra
using MetropolisHastings
using Plots, PrettyTables, ProgressMeter
using SpecialFunctions, Statistics

import Base.size, Base.map
import Statistics.mean, Statistics.cov, Statistics.cor
import DataFrames.stack
import Distributions.rand, Distributions.mode
import LinearAlgebra.cholesky