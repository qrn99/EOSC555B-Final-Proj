include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, PyCall, PyPlot

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
V_1(r) = r^(-12) - 2*r^(-6)

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
E_avg(X, f) = mean([f.(X[:, i]) for i = 1:size(X)[2]]) * sqrt(size(X)[2])

f = V_1

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

    fig, axs = PyPlot.subplots(length(K_1s),length(NN), sharex=true, sharey = true, figsize=(10, 10))
    # push!(plots, histogram(rand(Uniform(domain_lower, domain_upper), 500), bins = 20, title="Uniform Dist"))
    for i = eachindex(K_1s)
        K_1 = K_1s[i]
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

            target_x = range(domain_lower, domain_upper, length=500)
            training_flatten = reduce(vcat, X)
            test_flatten = reduce(vcat, XX_test)

            axs[i,t][:plot](target_x, f.(target_x), label="target")
            axs[i,t][:plot](training_flatten, f.(training_flatten), color="black", "o", alpha=0.2, markersize=1, label="train")
            axs[i,t][:plot](XX_test, yp, color="orange", linestyle="dashed", linewidth=2, label="prediction")
            axs[i,t].set_ylabel(L"K_1=%$K_1")
            axs[i,t].legend(loc="upper right", fontsize = "x-small")
        end
    end

    for ax in fig.get_axes()
        ax.label_outer()
    end
    for j in eachindex(NN)
        max_deg = NN[j]
        axs[length(K_1s), j].set_xlabel("Basis Size = $max_deg")
    end

    for a in axs
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect("equal")
    end

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(exp_dir*"/pp_inc_deg_[" * string(NN[1]) * "," * string(NN[end]) * "]" * "_K_1=[" * string(K_1s[1]) * "," * string(K_1s[end]) * "]" * "_order=$ord" * "_solver=$solver"*string(f), bbox_inches="tight")
    fig
    # plt
end

let
    Testing_func(X) = E_avg(X, f)
    P = Plots.plot(xaxis  = (:log, "Sample Size"),
                            yaxis  = (:log, "RMSE"), 
                            legend = :topright, 
                            size = (400, 400))
    for K_1 in K_1s
        error = zeros(length(NN))'
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
        end
        plot!(P, MM, error', lw=1, m=:auto, ms=3, label=L"K_1=%$K_1")
        # plot!(P, MM, 1.0 ./ sqrt.(MM), ls=:dash,)
    end
    Plots.savefig(P, exp_dir*"/RMSE_inc_deg_[" * string(NN[1]) * "," * string(NN[end]) * "]" * "_K_1=[" * string(K_1s[1]) * "," * string(K_1s[end]) * "]" * "_order=$ord" * "_solver=$solver"*string(f))
    P
end



