require("options.jl")
import OptionsMod.*

reposDir = EnvHash()["JuliaRepos"]

## Load JuliaData (from JD's init.jl)
load("$reposDir/JuliaData/src/datavec.jl")
load("$reposDir/JuliaData/src/index.jl")
load("$reposDir/JuliaData/src/namedarray.jl")
load("$reposDir/JuliaData/src/dataframe.jl")
load("$reposDir/JuliaData/src/formula.jl")
load("$reposDir/JuliaData/src/utils.jl")

## Load julia-svm
load("$reposDir/julia-svm/src/svm.jl")

df = DataFrame(quote
    y  = [18.,17,15,20,10,20,25,13,12]
    x1 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    x2 = [1, 1, 1, 2, 2, 2, 3, 3, 3]
end)

#glm(:(y ~ x1 + x2), df, Poisson())
svp = svm_problem(:(y ~ x1 + x2), df)
svparam = svmparameter("epsilon_svr", "rbf", int32(3),
                       1., 0., 40., 0.001,
                       1., 0.5, 
                       1., int32(1), int32(0))
model = svmtrain(svp, svparam)
X2 = rand(10, 2)
pred = svmpredict(model, X2)
