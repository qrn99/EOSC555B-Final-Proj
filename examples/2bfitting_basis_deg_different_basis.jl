include("../src/HelperFunctions.jl")
# include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

exp_dir = "results/2b1dfitting/" # result saving dir
mkpath(exp_dir)

f1(x) = abs(x)^3
f2(x) = 1/(1+8*x^2)

E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/sqrt(length(R)) for R in Rs ]

testSampleSize=100
adaptedTrainSize=200
test_uniform=false
rdf_bimodal=0.3

# K_N = 1
# K_N = 4
# K_N = 16
K_N = 64
noise=0

    let NN = [5, 7, 10, 13, 17, 21, 25, 30, 35, 40], MM = NN.^2
        
        f=f1
        # f=f2
        # Add poly when pack has it
        basis_choices = [HelperFunctions.cheb, HelperFunctions.legendre, "orthpoly"]

        distributions = ["uniform", "hole", "bimodal", "abs"]

        plots = []
                
        data_dst = HelperFunctions.generate_data_dst(rdf_bimodal, -0.8, 0.9, -1, 1; abs_width=0.8)

        # push!(plots, histogram(rand(1000)*2 .- 1, bins = 20, title="uniform distribution", label=""))

        Random.seed!(12216859)
        # for each in keys(data_dst)
        #     push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, title="$each distribution", label=""))
        # end

            for dst in distributions 
                if f == f1
                    P = plot(xaxis  = (:log, "Basis Size"),
                            yaxis  = (:log, "RMSE"), 
                            legend = :topright, 
                            title = L"f_1"* " with $dst distribution",
                            size = (300, 100))
                    plot!(P, NN[2:7], NN[2:7].^(-3), lw=2, c=10, ls=:dash, ms=3, label = L"J^{-3}")
                else
                    P = plot(xaxis  = ("Basis Size"),
                            yaxis  = (:log, "RMSE"), 
                            legend = :topright, 
                            title =  L"f_2"* " with $dst distribution",
                            size = (300, 100))
                    plot!(P, NN[2:7], exp.( - asinh(1/8^(1/2)) * NN[2:7]), lw=2, ls=:dash, c=10, ms=3, label = L"\exp(-\gamma J)")
                end
                    
                for basis_choice in basis_choices
                    err = HelperFunctions.LSQ_error_deg(NN, MM, K_N, testSampleSize, f, E, basis_choice, data_dst, dst; 
                                        noise=noise, test_uniform=test_uniform, adaptedTrainSize=adaptedTrainSize) 
                    plot!(P, NN, err', lw=1, m=:o, ms=3, label="$basis_choice")
                end
                push!(plots, P)
            end
        l = @layout [grid(2, 2)]
        # plot_title="Prediction Error Plot with Sufficient Data for Increasing Basis Size"
        p1 = plot(plots..., layout = l, size=(800, 700), margin = 3mm, plot_title=L"K_1=%$K_N")
        savefig(p1, exp_dir*"basic_model_RSME_inc_deg_K_1=$K_N" * "_noise=$noise" * "_test_unif=$test_uniform" * "_f = $f")
        p1
    end
