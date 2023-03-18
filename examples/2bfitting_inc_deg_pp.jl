include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

#save plot
exp_dir = "results/2b1dfitting/" # result saving dir
mkpath(exp_dir)

r_in = 0.85 # with envelope should be 0.0
r_nn = 1.0
r_cut = 3.0

# hyperparameter
α = 3   # morse coordinate change
p = 4   # envelope
q = 4   # prior smoothness, could be exp

# # pair potential
ϕ(r) = r^(-12) - 2*r^(-6)

# morse transform
# x1(r) = 1 / (1 + r / r_nn)

#Agnesis Transform
# 0.33 = a = (p-1)/(p+1)
x1(r) = 1.0 / (1.0 + 0.33*(r / r_nn)^2)

x_in = x1(0.0); x_cut = x1(r_cut) # Regularize til r=0
x(r) = 2 * (x1(r) - x_cut) / (x_in - x_cut) - 1

# envelope
env(r) = (r^(-p) - r_cut^(-p) + p * r_cut^(-p-1) * (r - r_cut)) * (r < r_cut)

f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3
# E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
# E_avg(X, f) = mean([f.(X[:, i]) for i = 1:size(X)[2]])
E_avg(Rs::Vector{Vector{Float64}}, f) = [ sum(f.(R))/sqrt(length(R)) for R in Rs ]


f = ϕ

ord = 1 #2b+3b, can access 3b only 
body_order = :TwoBody # 3b only

testSampleSize=400
test_uniform=true
adaptedTrainSize=testSampleSize
distribution=Uniform

domain_lower=r_in
domain_upper=r_cut+1

noise=0
# noise=1e-4

solver = :qr
# solver = :ard

NN = [5, 10, 20, 30]
MM = 10*NN.^2 .+ 50
K_1s = [1, 4, 16, 64]
# K_1s = [1, 4]
let
    Testing_func(X) = E_avg(X, f)
    plots = []
    # push!(plots, histogram(rand(Uniform(domain_lower, domain_upper), 500), bins = 20, title="Uniform Dist"))
    for K_1 in K_1s
        error = zeros(length(NN))'
        P = plot(xaxis  = (:log, "sample size", ),
                            yaxis  = (:log, "RMSE"), 
                            legend = :topright, 
                            size = (300, 100))
        for t = eachindex(NN)
            M = MM[t]
            max_degree = NN[t]

            poly = legendre_basis(max_degree, normalize = true)

            X = rand(distribution(domain_lower, domain_upper), (M, K_1))
            Y = Testing_func(X) .+ noise

            A_pure = designMatNB(x.(X), poly, max_degree, ord; body = body_order)

            sol_pure = solveLSQ(A_pure, Y; solver=solver)
                    
            XX_test = range(domain_lower, domain_upper, length=testSampleSize)

            A_test = predMatNB(x.(XX_test), poly, max_degree, ord; body = body_order)
            yp = A_test * sol_pure
            ground_yp = f.(XX_test)
            RMSE = norm(yp - ground_yp)/sqrt(M)

            println("Max Basis Deg: $max_degree, K_1: $K_1")
            println("Relative error: ", norm(yp - ground_yp)/norm(ground_yp))
            println("RMSE: ", RMSE)

            error[t] = RMSE

            target_x = range(domain_lower, domain_upper, length=500)
            p = plot(target_x, f.(target_x), c=1,
            #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
                        size = (1000, 800),
                        label = "target", xlabel=L"r", ylabel=L"V_1(r)", title=L"K_1=%$K_1"*", Basis Size = $max_degree")
            training_flatten = reduce(vcat, X)
            test_flatten = reduce(vcat, XX_test)
            plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, ma=0.008, label = "train")
            plot!(XX_test, yp, c=2, ls=:dash, lw=2, label = "prediction")
            push!(plots, p) 
        end
        plot!(P, MM, error', lw=1, m=:o, ms=3, label="K_1=$K_1")
        push!(plots, P)
    end
    l = @layout [grid(length(K_1s), length(NN)+1)]
        
    plt = plot(plots..., layout = l, size=(2500, 1300), margin=15mm)
    savefig(plt, exp_dir*"/pp_inc_deg_[" * string(NN[1]) * "," * string(NN[end]) * "]" * "_K_1=[" * string(K_1s[1]) * "," * string(K_1s[end]) * "]" * "_order=$ord" * "_solver=$solver")
    plt
end




