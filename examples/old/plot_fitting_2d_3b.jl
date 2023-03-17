include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

f1(x) = 1/(1+8*norm(x)^2)
f2(x) = abs(norm(x))^3

f1_V2(x, y) = x^2 + y^2
#f2_V2(X) = sum([X[:, i].^2 - 10 * cos.(2 * pi * X[:, i]) .+ 10 for i = 1:size(X)[2]])

# E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])

# 2 body avg energy
function E_avg(X, f)
    M, K_R, _ = size(X)
    b = zeros(M)
    for i = 1:M
        b[i] = mean([f(X[i, k, :]) for k = 1:K_R])
    end
    return b
end

# 3 body avg energy, f should be f(x, y) → R, X is of dize (M, K_R, 2)
function E_avg3b(X, f)
    M, K_R, _ = size(X)
    b = zeros(M)
    for i = 1:M
        b[i] = mean([f(X[i, j, 1], X[i, k, 2]) for j = 1:K_R for k = 1:K_R if (j != k)]) # taking mean of all paris of 3b energy with i ≂̸ j
    end
    return b
end


M = 500
max_deg_poly = 20
max_deg_exp = 5
ord = 2 # this variable has no use now
body_order = :ThreeBody

testSampleSize=50
test_uniform=true
distribution=Uniform

# avoid using distance transform by restricting range
domain_lower=-sqrt(2)/2
domain_upper= sqrt(2)/2 
K_R = 10
K_R_test = 4

noise=0
# noise=1e-4

solver = :qr

f = f1_V2
Testing_func(X) = E_avg3b(X, f)
poly = legendre_basis(max_deg_poly, normalize = true)

train = rand(distribution(domain_lower, domain_upper), (M, K_R, 2))
@show size(train)
#@show size([train[k, :, :] for k = 1:size(X)[1]])
#@show size(X)
Y = Testing_func(train)

A_pure, spec = designMatNB2D(train, poly, max_deg_poly, max_deg_exp, ord; body= body_order)
sol_pure = solveLSQ(A_pure, Y; solver=solver)

# XX_test = range(domain_lower, domain_upper, length=testSampleSize)

XX_test = rand(distribution(domain_lower, domain_upper), (testSampleSize, K_R_test, 2))
#XX_test = XX_test[sortperm(XX_test[:, 1]), :]

# XX_test_r1 = sort(rand(distribution(domain_lower, domain_upper), M))
# XX_test_r2 = sort(rand(distribution(domain_lower, domain_upper), M))

# A_test = predMatNB(XX_test, poly, max_degree, ord; body = body_order)
A_test, spec_test = designMatNB2D(XX_test, poly, max_deg_poly, max_deg_exp, ord; body = body_order)
@show maximum(A_pure)
@show maximum(A_test)
yp = A_test * sol_pure
@show maximum(sol_pure)
@show maximum(yp)
ground_yp = Testing_func(XX_test)

@show maximum(ground_yp)
println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
println("RMSE: ", sqrt(norm(yp - ground_yp)/testSampleSize))

# target_x = range(domain_lower, domain_upper, length=500)
# target_y = range(domain_lower, domain_upper, length=500)
# target_z = f1_V2.(target_x, target_y)
# # V2(hcat(target_x, target_y)
# plotly();
# p = plot(target_x, target_y, f, st=:surface,
# #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
#             size = (1000, 800), alpha = 0.5,
#             label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
# training_flatten = reduce(vcat, X)
# test_flatten = reduce(vcat, XX_test)
# scatter!(X[:, 1], X[:, 2], Y, seriestype=:scatter, m=:o, markercolor = :red, ms=1.5, label = "ground truth")
# scatter!(XX_test[:,1], XX_test[:, 2], yp, seriestype=:scatter, m=:o, markercolor=:green, ms=1, label = "prediction")
# p

