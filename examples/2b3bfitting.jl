include("../src/utils.jl")

# -------------------------------------------------------testing functionss------------------------------------------#
function Rastrigin(X)
    return sum([X[:, i].^2 - 10 * cos.(2 * pi * X[:, i]) .+ 10 for i = 1:size(X)[2]])
end
 

# -------------------------------------------------------testing configs------------------------------------------#
Random.seed!(1255534)
max_degree = 15 # max_degree of polynomial
exp_dir = "results/test_result_3b_Rastrigin_dnumsam_samedegree/" # result saving dir
mkpath(exp_dir)
distribution = Uniform # determinated by type of basis used
solver = "qr" # solver, currently support qr and ARD
num_sam_list = [10000] # number of data used in training
num_test = 2000 # number of data used in test, will be drawn from the same distribution as training
aa = num_sam_list[1]
bb = num_sam_list[end]
ord = 1 # ord = body - 1
K_R = 5 # number of election within a neighbourhood
poly = legendre_basis(max_degree, normalize = true) # polynomial basis used
Testing_func = Rastrigin # testing function for evaluating basis
plotting = true # decide to save result or not

# -------------------------------------------------------basic constructions------------------------------------------#
NN = get_NN(max_degree, ord)
NN2b = NN[length.(NN) .== 1] # for analysis
NN3b = NN[length.(NN) .== 2]

pure_err = []
cond_num_2b_pure = []
cond_num_3b_pure = []

# -------------------------------------------------------testing loop------------------------------------------#
for num_sam in num_sam_list
    pure_err_curr = 0
    cond_num_2b_pure_curr = 0
    cond_num_3b_pure_curr = 0
    
    for i = 1:10
       num_sam = Integer(num_sam)
       println("----------------------------------------------")
       println("current number of sample points: ", num_sam)
       path = exp_dir * "maxdeg="*string(max_degree)*"/num_sam="*string(num_sam)*"/"
       mkpath(path)
       X = rand(distribution(-1, 1), (num_sam, K_R))
 
 
       # initialize design matrix
       A_pure = zeros(num_sam, length(NN))
       B = Testing_func(X)

       # for evaluating ground truth
       poly_list = [poly(X[:, i]) for i = 1:K_R]
 
       for i = 1:length(NN2b)
          nn = NN2b[i]
          A_pure[:, i] = sum([PX1[:, nn] for PX1 in poly_list])
       end
       
        for i = 1:length(NN3b)
            nn, mm = NN3b[i]
            A_pure[:, length(NN2b) + i] = sum([PX1[:, nn] .* PX2[:, mm] for PX1 in poly_list for PX2 in poly_list if PX1 != PX2])
        end

 
       if plotting == true
        # check orthogonality
        heatmap(A_pure[:,1:length(NN2b)]' * A_pure[:,1:length(NN2b)] , aspect_ratio=1)
        savefig(exp_dir * "check_A_pure2b")
        if length(NN3b) > 0
            heatmap(A_pure[:,length(NN2b) + 1:end]' * A_pure[:,length(NN2b) + 1:end] , aspect_ratio=1)
            savefig(exp_dir * "check_A_pure3b")
        end
       end
       

       if solver == "qr"
          # solve the problem with qr
          LL = length(NN)
          λ = 0.1
          sol_pure = qr(vcat(A_pure, λ * I(LL) + zeros(LL, LL))) \ vcat(B, zeros(LL))
       elseif solver == "ARD"
       # solve the problem with ARD       
          ARD = pyimport("sklearn.linear_model")["ARDRegression"]
          clf = ARD()
          sol_pure = clf.fit(A_pure, B).coef_
       end
       
       # testing
       XX_test = rand(distribution(-1, 1), (num_test, K_R))
       A_test = zeros(num_test, length(NN))

       poly_list_test = [poly(XX_test[:, i]) for i = 1:K_R]

       for i = 1:length(NN2b)
        nn = NN2b[i]
        A_test[:, i] = sum([PX1[:, nn] for PX1 in poly_list_test])
        end

        for i = 1:length(NN3b)
            nn, mm = NN3b[i]
            A_test[:, length(NN2b) + i] = sum([PX1[:, nn] .* PX2[:, mm] for PX1 in poly_list_test for PX2 in poly_list_test if PX1 != PX2])
        end
       zs_pure = A_test * sol_pure
       ground_zs = Testing_func(XX_test)
 
       println("relative error of pure basis: ", norm(zs_pure - ground_zs)/norm(ground_zs))
       
       pure_err_curr += norm(zs_pure - ground_zs)/norm(ground_zs)
 
       cond_num_2b_pure_curr += cond(A_pure[:, 1:max_degree])
       @show cond_num_2b_pure_curr
       if length(NN3b) > 0
           cond_num_3b_pure_curr += cond(A_pure[:, length(NN2b) + 1: end])
       end
    end

    push!(pure_err, pure_err_curr / 10)
    push!(cond_num_2b_pure, cond_num_2b_pure_curr / 10) 
    push!(cond_num_3b_pure, cond_num_3b_pure_curr/ 10)
end

# -------------------------------------------------------result saving------------------------------------------#

if plotting == true
    p1 = plot(num_sam_list, pure_err, label = "pure", xlabel = "num sam", ylabel = "rel err", yscale = :log10)
    savefig(p1, exp_dir*"/checkerror_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]")

    p2 = plot(num_sam_list, cond_num_2b_pure, label = "2b", xlabel = "num sam", ylabel = "Coniditional number")
    savefig(p2, exp_dir*"/checkcond_2b_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]")

    p4 = plot(num_sam_list, cond_num_3b_pure, label = "3b", xlabel = "num sam", ylabel = "Coniditional number")
    savefig(p4, exp_dir*"/checkcond_3b_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]")

    CSV.write(exp_dir*"/err_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]" * ".csv", (pure_err = pure_err)) 
    CSV.write(exp_dir*"/cond_sep_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]" * ".csv", (cond_num_2b_pure = cond_num_2b_pure, cond_num_3b_pure = cond_num_3b_pure)) 
end

 

