include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

#save plot
exp_dir = "results/2b1dfitting/" # result saving dir
mkpath(exp_dir)

NN = [5, 10, 20, 30]
MM = 10*NN.^2 .+ 50
K_1s = [1, 4, 16, 64]
# K_1s = [1, 4]


# constant
r_in = 0.85 # with envelope should be 0.0
r_nn = 1.0
r_cut = 3.0

# hyperparameter
α = 3   # morse coordinate change
p = 4   # envelope
q = 4   # prior smoothness, could be exp

# pair potential
ϕ(r) = r^(-12) - 2*r^(-6)

E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/sqrt(length(R)) for R in Rs ]
E(f, R::Vector{Float64}) = sum(f.(R)) /sqrt(length(R))

# morse transform
# x1(r) = 1 / (1 + r / r_nn)

#Agnesis Transform
# 0.33 = a = (p-1)/(p+1)
x1(r) = 1.0 / (1.0 + 0.33*(r / r_nn)^2)

x_in = x1(0.0); x_cut = x1(r_cut) # Regularize til r=0
x(r) = 2 * (x1(r) - x_cut) / (x_in - x_cut) - 1

# envelope
env(r) = (r^(-p) - r_cut^(-p) + p * r_cut^(-p-1) * (r - r_cut)) * (r < r_cut)

# weights to downgrade high energys
W_R(R, target_fnc, averge_est_fnc) = 1/( max(length(R), averge_est_fnc(target_fnc, R))^2 )

function design_matrix_pp(rs, basis, N)
    A = zeros(ComplexF64, length(rs), N)
    for (i, rr) in enumerate(rs)
      A[i, :] = sum(evaluate(basis, x(r)) * env(r) for r in rr) / sqrt(length(rr))
    end 
    return A
end

function predict_pp(r::Float64, basis, μ)
   B = evaluate(basis, x(r)) * env(r)
   val = real(sum(μ .* B))
   return val
end

function predict_pp(rr, args...)
    vals = Float64[]
    for r in rr
        v = predict_pp(r, args...)
        push!(vals, v)
    end
    return vals
end

function lr_pp(averge_est_fnc, target_fnc, train, basis, N; noise=0.0, restraint=true, weight_blend=1e-10)
    Ws = W_R.(train, target_fnc, averge_est_fnc) 
#     @show Ws
#     @show averge_est_fnc(target_fnc, train)
    Y = Ws .* averge_est_fnc(target_fnc, train) .+ noise
    Ψ = design_matrix_pp(train, basis, N)
    
    for (i, rr) in enumerate(train)
       Ψ[i, :] = Ws[i] * Ψ[i, :]
    end
    
    if restraint
      Ψ_blend = weight_blend*evaluate(basis, 0.0)'
      Y_blend = [1.0]
      Ψ = [Ψ; Ψ_blend]
      Y = [Y; Y_blend]
    end
    
    Q,R = qr(Ψ)
    μ = R \ (Matrix(Q)'*Y)  
    
#     μ = Ψ \ Y 
    return μ
end

function LSQ_error_pp(NN, MM, K_1, testSampleSize, target_fn, averge_est_fn, basis, data_dst, dst; 
        noise=0.0, test_uniform=false, restraint=true, weight_blend=1e-10, r_in=r_in, r_cut=r_cut)
    err = zeros(1, length(NN))

    for i=1:length(NN)
        N = NN[i]
        M = MM[i]

        train, test = HelperFunctions.generate_data(M, K_1, testSampleSize, data_dst, dst; r_in = r_in, r_cut = r_cut)

        #get parameter
        μ = lr_pp(averge_est_fn, target_fn, train, basis, N; noise=noise, restraint=restraint, weight_blend=weight_blend)

        if test_uniform
            test = sort(collect(Iterators.flatten([ rand(K_1)*(r_cut-r_in) .+ r_in for _=1:testSampleSize ])))
        else
            test = sort(collect(Iterators.flatten(test)))
        end
        
        yp = target_fn.(test)
        y = predict_pp(test, basis, μ)
        
        for j=1:length(y)
            if yp[j] >= 1
                yp[j]=0
                y[j]=0
#                 @show j, "changed"
            end
        end

        err[i] = norm(yp - y, 2)/sqrt(length(test))
    end
    
    return err
end

function LSQ_error_deg_pp(NN, MM, K_1, testSampleSize, target_fn, averge_est_fn, basis_choice, data_dst, dst; 
        noise=0.0, test_uniform=false, restraint=true, weight_blend=1e-10, adaptedTrainSize=100, r_in=r_in, r_cut=r_cut)
    err = zeros(1, length(NN))

    for i=1:length(NN)
        N = NN[i]
        M = MM[i]
        
        train, test = HelperFunctions.generate_data(M, K_1, testSampleSize, data_dst, dst; r_in = r_in, r_cut = r_cut)
        
        basis = HelperFunctions.get_basis(basis_choice, N, K_1, data_dst, dst; adaptedTrainSize)

        #get parameter
        μ = lr_pp(averge_est_fn, target_fn, train, basis, N; noise=noise, restraint=restraint, weight_blend=weight_blend)

        if test_uniform
            test = sort(collect(Iterators.flatten([ rand(K_1)*(r_cut-r_in) .+ r_in for _=1:testSampleSize ])))
        else
            test = sort(collect(Iterators.flatten(test)))
        end
        
        yp = target_fn.(test)
        y = predict_pp(test, basis, μ)
        
        for j=1:length(y)
            if yp[j] >= 1
                yp[j]=0
                y[j]=0
            end
        end

        err[i] = norm(yp - y, 2)/sqrt(length(test))
    end
    
    return err
end

restraint=true
weight_blend=1e-8
rdf_bimodal=0.22
let NN = [5, 7, 10, 13, 17, 21, 25, 30, 35, 40], MM = NN.^2, fs=[ϕ]

    # Add poly when pack has it
    basis_choices = [HelperFunctions.legendre]

    # distributions = ["uniform", "hole", "bimodal", "abs"]
    distributions = ["uniform", "hole"]

    plots = []
            
    data_dst = HelperFunctions.generate_data_dst(rdf_bimodal, r_in, r_cut, 0.1, 5)

    # push!(plots, histogram(rand(1000)*(r_cut-r_in) .+ r_in, bins = 20, title="Uniform Dist"))

    # Random.seed!(12216859)
    # for each in keys(data_dst)
    #     push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, title="$each Dist"))
    # end

    for f in fs
        for dst in distributions 
            P = plot(xaxis  = ("basis size"),
                    yaxis  = (:log, "RMSE"), 
                    legend = :outerbottomright, 
                    title = L"V_1" * " with $dst distribution",
                    size = (300, 100))
            plot!(P, NN[1:4], exp.( -0.61 * NN[1:4]), lw=2, ls=:dash, c=10, ms=3, label = L"\exp(-\gamma J),")
            # plot!(P, NN[2:5],  NN[2:5] .^ (-1), lw=2, ls=:dash, c=10, ms=3, label = L"J^{-\infty}")
            
            for K_1 in K_1s    
                for basis_choice in basis_choices
                    err = LSQ_error_deg_pp(NN, MM, K_1, testSampleSize, f, E, basis_choice, data_dst, dst; 
                                        noise=noise, restraint=restraint, weight_blend=weight_blend, test_uniform=test_uniform, adaptedTrainSize=adaptedTrainSize) 
                    plot!(P, NN, err', lw=1, m=:o, ms=3, label=L"K_1=%$K_1")
                end
            end
            push!(plots, P)
        end
    end
    # l = @layout [grid(length(distributions), length(fs))]
    l = @layout [grid(length(fs), length(distributions))]
    # plot_title="Prediction Error Plot with Sufficient Data for Increasing Basis Size"
    p1 = plot(plots..., layout = l, size=(1000, 400), margin = 10mm)
    # p1 = plot(plots..., layout = l, size=(2200, 500), margin = 15mm)
    savefig(p1, exp_dir*"basic_pp_RSME_basis_inc_basis_no="*string(length(basis_choices)) * "_dst_no="*string(length(distributions)))
    p1
    end
    
    let fs=[ϕ]
        restraint=true
        weight_blend=1e-3
        rdf_bimodal=0.22
        
        N = 20
        M = N^2
        K_1 = 1
        basis_choices = [HelperFunctions.cheb, HelperFunctions.legendre, "orthpoly"]
    
        distributions = ["uniform", "hole", "bimodal", "abs"]
        
        plots = []
        
        data_dst = HelperFunctions.generate_data_dst(rdf_bimodal, r_in, r_cut, 0.1, 5)
    
        push!(plots, histogram(rand(1000)*(r_cut-r_in) .+ r_in, bins = 20, title="Uniform Dist"))
    
        Random.seed!(12216859)
        for each in keys(data_dst)
            push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, title="$each Dist"))
        end
    
        for f in fs
            for basis_choice in basis_choices
                for dst in distributions 
                
                train, test = HelperFunctions.generate_data(M, K_1, testSampleSize, data_dst, dst; r_in = r_in, r_cut = r_cut)
        
                training_flatten = sort(collect(Iterators.flatten(train)))
                xp_pp_test = range(0.1, 5, length=300)
#                 xp_pp_test = sort(collect(Iterators.flatten(test)))
                
                basis = HelperFunctions.get_basis(basis_choice, N, K_1, data_dst, dst; adaptedTrainSize=adaptedTrainSize)

                μ = lr_pp(E, f, train, basis, N; noise=noise,  restraint=restraint, weight_blend=weight_blend)

                y_pp = predict_pp(xp_pp_test, basis, μ)
                
                target_x = range(0.5, 3.5, length=200)
                p = plot(target_x, f.(target_x), c=1, ylim=[-1.5, 2.5],
                    label = "target", xlabel="x", ylabel="pair potential", title="$basis_choice with $dst distribution")

                plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, ma=0.08, label = "")
                plot!(xp_pp_test, y_pp, c=2, label = "prediction")
                    
                push!(plots, p)
                    
                end
            end
        end
        l = @layout [grid(length(fs)+length(basis_choices)+1, length(distributions))]
    
        p1 = plot(plots..., layout = l, size=(2500, 1500), margin = 5mm)
        savefig(p1, exp_dir*"basic_pp_deg=$N" * "_M=$M" * "_J=$K_1" * "_noise=$noise" * ".png")
        p1
    end