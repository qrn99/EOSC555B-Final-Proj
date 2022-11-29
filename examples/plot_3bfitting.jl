include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3
E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])

M = 200
max_degree = 20
ord = 2 #2b+3b, can access 3b only 
body_order = :ThreeBody

testSampleSize=400
test_uniform=true
adaptedTrainSize=testSampleSize

domain_lower=-1
domain_upper=1
K_R = 4
noise=1e-4

solver = :qr

f = f2
Testing_func(X) = E_avg(X, f)
poly = legendre_basis(max_degree, normalize = true)

X = rand(distribution(domain_lower, domain_upper), (M, K_R))
Y = Testing_func(X)

A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)

if solver == :qr
    # solve the problem with qr
    LL = size(A_pure)[2]
    λ = 0.1
    sol_pure = qr(vcat(A_pure, λ * I(LL) + zeros(LL, LL))) \ vcat(Y, zeros(LL))
elseif solver == :ard
# solve the problem with ARD       
    ARD = pyimport("sklearn.linear_model")["ARDRegression"]
    clf = ARD()
    sol_pure = clf.fit(A_pure, Y).coef_
end
        

XX_test = range(-1, 1, length=testSampleSize)

A_test = predMatNB(XX_test, poly, max_degree, ord; body = body_order)
yp = A_test * sol_pure
ground_yp = f.(XX_test)

println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
println("RMSE: ", norm(yp - ground_yp)/sqrt(M))

target_x = range(domain_lower, domain_upper, length=500)
p = plot(target_x, f.(target_x), c=1,
#                         xlim=[-1.1, 1.1], ylim=[-1, 2],
            size = (1000, 800),
            label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
training_flatten = reduce(vcat, X)
test_flatten = reduce(vcat, XX_test)
plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, label = "")
plot!(XX_test, yp, c=2, ls=:dash, label = "prediction")
p
