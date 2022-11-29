include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact, PyCall
##
f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3
# E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
E_avg(X, f) = sum([f.(X[:, i])/length(size(X)[2]) for i = 1:size(X)[2]])

##
M = 200
max_degree = 10
ord = 2 #2b+3b
body_order = :TwoBodyThreeBody

testSampleSize=400
test_uniform=true
distribution=Uniform

domain_lower=-1
domain_upper=1
K_R = 4

noise=0
# noise=1e-4

solver = :ard

Testing_func(X) = E_avg(X, f1) + E_avg(X, f2)
poly = legendre_basis(max_degree, normalize = true)

X = rand(distribution(domain_lower, domain_upper), (M, K_R))
Y = Testing_func(X) .+ noise

A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)

sol_pure = solveLSQ(A_pure, Y; solver=solver) 

XX_test = range(domain_lower, domain_upper, length=testSampleSize)

A_test = predMatNB(XX_test, poly, max_degree, ord; body = body_order)
yp = A_test * sol_pure
ground_yp = f1.(XX_test) + f2.(XX_test)

println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
println("RMSE: ", norm(yp - ground_yp)/sqrt(M))

target_x = range(domain_lower, domain_upper, length=500)
p = plot(target_x, f1.(target_x)+f2.(target_x), c=1,
#                         xlim=[-1.1, 1.1], ylim=[-1, 2],
            size = (1000, 800),
            label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
training_flatten = reduce(vcat, X)
test_flatten = reduce(vcat, XX_test)
plot!(training_flatten, f1.(training_flatten)+f2.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, label = "")
plot!(XX_test, yp, c=2, ls=:dash, label = "prediction")
p
##