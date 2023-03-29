include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall, PyPlot

exp_dir = "results/2b1dfitting/" # result saving dir
mkpath(exp_dir)

f_1(x) = 1/(1+8*x^2)
f_2(x) = abs(x)^3
# E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
# E_avg(X, f) = sum([f.(X[:, i])/sqrt(length(size(X)[2])) for i = 1:size(X)[2]])
E_avg(X, f) = mean([f.(X[:, i]) for i = 1:size(X)[2]]) * sqrt(size(X)[2])

f = f_1

ord = 1 #2b+3b, can access 3b only 
body_order = :TwoBody # 3b only

testSampleSize=400
test_uniform=true
adaptedTrainSize=testSampleSize
distribution=Uniform

domain_lower=-1
domain_upper=1

noise=0
# noise=1e-4

solver = :qr
# solver = :ard

NN = [5, 10, 20, 30]
MM = 10*NN.^2 .+ 50
# MM = Int.(floor.(log.(NN) .* NN))
K_1s = [1, 4, 16, 64]
# K_1s = [1, 4]

let fs = [f_1, f_2]
    for f in fs
        Testing_func(X) = E_avg(X, f)
        plots = []

        fig, axs = PyPlot.subplots(length(K_1s),length(NN), sharex=true, sharey = true, figsize=(10, 6))
        # push!(plots, histogram(rand(Uniform(domain_lower, domain_upper), 500), bins = 20, title="Uniform Dist"))
        for i = eachindex(K_1s)
            K_1 = K_1s[i]
            for t = eachindex(NN)
                M = MM[t]
                max_degree = NN[t]

                poly = legendre_basis(max_degree, normalize = true)

                X = rand(distribution(domain_lower, domain_upper), (M, K_1))
                Y = Testing_func(X) .+ noise

                A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)
                sol_pure = solveLSQ(A_pure, Y; solver=solver)
                XX_test = reshape(range(domain_lower, domain_upper, length=testSampleSize), (testSampleSize, 1))
                #XX_test = rand(distribution(domain_lower, domain_upper), (testSampleSize, 1))

                A_test = designMatNB(XX_test, poly, max_degree, ord; body = body_order)

                yp = A_test * sol_pure
                ground_yp = Testing_func(XX_test)
                RMSE = norm(yp - ground_yp)/sqrt(M)

                println("Max Basis Deg: $max_degree")
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
    end
end

let fs = [f_1, f_2]
    plots = []
    for f in fs
        Testing_func(X) = E_avg(X, f)
        P = Plots.plot(xaxis  = (:log, "Sample Size"),
                                yaxis  = (:log, "RMSE"), 
                                legend = :topright, 
                                size = (600, 500),
                                title=L"%$f")
        
        for K_1 in K_1s
            error = zeros(length(NN))'
            for t = eachindex(NN)
                M = MM[t]
                max_degree = NN[t]

                poly = legendre_basis(max_degree, normalize = true)

                X = rand(distribution(domain_lower, domain_upper), (M, K_1))
                Y = Testing_func(X) .+ noise

                A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)
                sol_pure = solveLSQ(A_pure, Y; solver=solver)
                XX_test = reshape(range(domain_lower, domain_upper, length=testSampleSize), (testSampleSize, 1))
                #XX_test = rand(distribution(domain_lower, domain_upper), (testSampleSize, 1))

                A_test = designMatNB(XX_test, poly, max_degree, ord; body = body_order)

                yp = A_test * sol_pure
                ground_yp = Testing_func(XX_test)
                RMSE = norm(yp - ground_yp)/sqrt(M)

                println("Max Basis Deg: $max_degree")
                println("Relative error: ", norm(yp - ground_yp)/norm(ground_yp))
                println("K_1 = $K_1, $f RMSE: ", RMSE)
                error[t] = RMSE
            end
            plot!(P, MM, error', lw=1, m=:auto, ms=3, label=L"K_1=%$K_1")
            # plot!(P, MM, 1.0 ./ sqrt.(MM), ls=:dash,)
        end
        push!(plots, P)
    end
    plt = Plots.plot(plots..., layout = (1, 2), size=(800, 400), margin=6mm)
    Plots.savefig(plt, exp_dir*"/RMSE_inc_deg_[" * string(NN[1]) * "," * string(NN[end]) * "]" * "_K_1=[" * string(K_1s[1]) * "," * string(K_1s[end]) * "]" * "_order=$ord" * "_solver=$solver"*string(fs))
    plt
end



