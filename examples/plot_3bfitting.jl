include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3

f1_V2(x, y) = x^2 + y^2
f2_V2(X) = sum([X[:, i].^2 - 10 * cos.(2 * pi * X[:, i]) .+ 10 for i = 1:size(X)[2]])

# E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
E_avg(X, f) = sum([f.(X[:, i])/length(size(X)[2]) for i = 1:size(X)[2]])

V2(X, f_V2) = [f_V2(X[j, :][1], X[j, :][2]) for j = 1:size(X)[1]]

# function E_avg3b(X, f)
#     for i = eachindex()
#     return
# end

M = 2000
max_degree = 15
ord = 2 #2b+3b, can access 3b only 
body_order = :ThreeBody

testSampleSize=50
test_uniform=true
distribution=Uniform

domain_lower=-1
domain_upper=1
K_R = 4

noise=0
# noise=1e-4

solver = :qr

f = f1_V2
Testing_func(X) = V2(X, f)
poly = legendre_basis(max_degree, normalize = true)

X = rand(distribution(domain_lower, domain_upper), (M, K_R))
Y = Testing_func(X)

A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)
@show cond(A_pure)
sol_pure = solveLSQ(A_pure, Y; solver=solver)

# XX_test = range(domain_lower, domain_upper, length=testSampleSize)

XX_test = rand(distribution(domain_lower, domain_upper), (testSampleSize, K_R))
XX_test = XX_test[sortperm(XX_test[:, 1]), :]

# XX_test_r1 = sort(rand(distribution(domain_lower, domain_upper), M))
# XX_test_r2 = sort(rand(distribution(domain_lower, domain_upper), M))

# A_test = predMatNB(XX_test, poly, max_degree, ord; body = body_order)
A_test = designMatNB(XX_test, poly, max_degree, ord; body = body_order)
yp = A_test * sol_pure
ground_yp = V2(XX_test, f)

println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
println("RMSE: ", sqrt(norm(yp - ground_yp)/testSampleSize))

target_x = range(domain_lower, domain_upper, length=500)
target_y = range(domain_lower, domain_upper, length=500)
target_z = f1_V2.(target_x, target_y)
# V2(hcat(target_x, target_y)
#plotly();
p = plot(target_x, target_y, f, st=:surface,
#                         xlim=[-1.1, 1.1], ylim=[-1, 2],
            size = (1000, 800), alpha = 0.5,
            label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
training_flatten = reduce(vcat, X)
test_flatten = reduce(vcat, XX_test)
scatter!(X[:, 1], X[:, 2], Y, seriestype=:scatter, m=:o, markercolor = :red, ms=1.5, label = "ground truth")
scatter!(XX_test[:,1], XX_test[:, 2], yp, seriestype=:scatter, m=:o, markercolor=:green, ms=1, label = "prediction")
p

