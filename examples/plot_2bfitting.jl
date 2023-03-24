include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact

##
f1(x) = abs(x)^3
f2(x) = 1/(1+8*x^2)
E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/sqrt(length(R)) for R in Rs ]
exp_dir = "results/2bfitting/" # result saving dir
mkpath(exp_dir)
# ##
# # @manipulate for BasisDeg = [5, 10, 20, 40, 80],  M = [40, 80, 160, 200], 
# #     testSampleSize=[20, 50, 100], adaptedTrainSize=[100, 150, 200], 
# #     test_uniform=[false, true],
# #     rdf_bimodal = range(0.15, 0.3, 16), K_1 = [1, 4, 16, 64], noise=[0.0, 1e-4, 1e-3] #, 1e-2, 1e-1]

BasisDeg=20

num_data = 100
K_1 = 10

M = Int(floor(num_data / K_1))

testSampleSize=400
test_uniform=true
adaptedTrainSize=testSampleSize

domain_lower=-1
domain_upper=1
# noise=1e-4
noise=0

let fs=[f1, f2]
# let fs=[f2]
# let fs=[f1]
    N = BasisDeg

    # Add poly when pack has it
    basis_choices = [HelperFunctions.cheb, HelperFunctions.legendre, "orthpoly"]

    distributions = ["uniform", "hole", "bimodal", "abs"]

    plots = []

    data_dst = HelperFunctions.generate_data_dst(0.3, -0.8, 0.9, domain_lower, domain_upper; abs_width=0.8)

    push!(plots, histogram(rand(1000)*2 .- 1, bins = 20, title="uniform distribtuion", legend = false))

    Random.seed!(12216859)
    for each in keys(data_dst)
        push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, title="$each distribtuion", legend = false))
    end

    for f in fs
        for basis_choice in basis_choices
            for dst in distributions 
                train, test = HelperFunctions.generate_data(M, K_1, testSampleSize, data_dst, dst)
                # @show size(train)
                training_flatten = sort(collect(Iterators.flatten(train)))
    #                     xp_pp_test = sort(collect(Iterators.flatten(test)))
                #@show size(training_flatten)
                xp_pp_test = range(domain_lower, domain_upper, length=testSampleSize)
                basis = HelperFunctions.get_basis(basis_choice, BasisDeg, K_1, data_dst, dst; adaptedTrainSize=adaptedTrainSize)
                
                μ = HelperFunctions.lr(E, f, train, basis, N; noise=noise)
                yp = HelperFunctions.predict(xp_pp_test, basis, μ)
                target_x = range(-1, 1, length=200)
                p = plot(target_x, f.(target_x), c=1, lw=2,
    #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
                    label = "target", xlabel=L"x", ylabel=L"f(x)", title="$basis_choice with $dst distribtuion")

                plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, ma=0.08, label = "train")
                plot!(xp_pp_test, yp, c=2, lw=2, ls=:dash, label = "prediction")
                push!(plots, p)
            end
        end
    end
    l = @layout [grid(length(fs)+length(basis_choices)+2, length(distributions))]
    PL = plot(plots..., layout = l, margin = 5mm, size=(2600, 1500))
    # plot_title="J=$BasisDeg, K_1 = $K_1, noise=$noise, test_uniform=$test_uniform"
    savefig(PL, exp_dir*"2b_basic_with_train_plots"*"_J=$BasisDeg"*"_K_1=$K_1"* "_total_data=$num_data" * "_M=$M" *"_f="*string(fs))
    # savefig(PL, exp_dir*"2b_basic_plots_KN=" * string(K_1)*"f=$fs")
    PL
end

# condition number 
let 
    N = BasisDeg
    let K_1s = [1, 4, 16, 64], Ms = 2 .^(7:13)
        testSampleSize = 10 # does not matter for design matrix
        # Add poly when pack has it
        basis_choices = [HelperFunctions.cheb, HelperFunctions.legendre, "orthpoly"]
    
        distributions = ["uniform"]
    
        plots = []
    
        data_dst = HelperFunctions.generate_data_dst(0.3, -0.8, 0.9, domain_lower, domain_upper; abs_width=0.8)

#         push!(plots, histogram(rand(1000)*2 .- 1, bins = 20, title="Uniform Dist"))
    
#         Random.seed!(12216859)
#         for each in keys(data_dst)
#             push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, title="$each Dist"))
#         end
        for K_1 in K_1s
            if K_1 > 4
                legend_pos = :bottomright
            else
                legend_pos = :topright
            end
            for dst in distributions 
                P = plot(xaxis  = (:log, "Sample Size (M)"),
                            yaxis  = (:log, L"\kappa(A^TA)"), 
                            legend = legend_pos, 
                            title = L"K_1=%$K_1",
                            size = (200, 200),
                            link=:all)
                plot!(P, Ms, 1 .^(7:13), c=1, ls=:dash, label="")

                for basis_choice in basis_choices
        #                     basis = get_basis(basis_choice, N, K_1, data_dst, dst; adaptedTrainSize=adaptedTrainSize)
                    conds = zeros(1, length(Ms))
                    for i in 1:length(Ms)
                        M = Ms[i]
                        # give sufficient data for adapted basis
                        basis = HelperFunctions.get_basis(basis_choice, N, K_1, data_dst, dst; adaptedTrainSize=M)
                        rs, _ = HelperFunctions.generate_data(M, K_1, testSampleSize, data_dst, dst)
                        A = HelperFunctions.design_matrix(rs, basis, N)
                        conds[i] = cond(A'A)
                    end
                    plot!(P, Ms, conds', xscale=:log10, yscale=:log10, shape=:auto, alpha=0.8, label="$basis_choice", link=:all)
                end
                push!(plots, P)
            end
        end
        l = @layout [grid(2,2)]
        # plot_title="Condition Number of the Gram Matrix with Basis Size=$N"
        p1 = plot(plots..., layout = l, size=(800, 700), margin = 3mm, link=:all)
        savefig(p1, exp_dir*"basic_model_cond_no_deg=$N"*"dst="*string(distributions))
        p1
    end
end
##

x = range(0,1,101)
p1 = plot(x,[sin.(x) exp.(x)],layout=(2,1),link=:all,label=nothing);
p2 = plot(x,[cos.(x) tan.(1.5x)],layout=(2,1),label=nothing);
plot([p1,p2]...,layout=(1,2))


#### Plot Data Distribution ####
# let fs=[f1, f2]
#     N = 20
#     # Add poly when pack has it
#     basis_choices = [HelperFunctions.cheb, HelperFunctions.legendre, "orthpoly"]

#     distributions = ["uniform", "hole", "bimodal", "abs"]

#     plots = []

#     data_dst = HelperFunctions.generate_data_dst(0.3, -0.8, 0.9, domain_lower, domain_upper; abs_width=0.8)

#     push!(plots, histogram(rand(1000)*2 .- 1, bins = 50, xaxis=L"x", yaxis="occurrence", title="uniform distribtuion", legend = false))

#     Random.seed!(12216859)
#     for each in keys(data_dst)
#         push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, xaxis=L"x", title="$each distribtuion", legend = false))
#     end
#     l = @layout [grid(1, length(distributions))]
#     PL = plot(plots..., layout = l, margin = 5mm, size=(1000, 200))
#     # plot_title="J=$BasisDeg, K_1 = $K_1, noise=$noise, test_uniform=$test_uniform"
#     savefig(PL, exp_dir*"data_dst")
#     PL
# end