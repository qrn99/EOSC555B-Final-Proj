# include("../ACEcore.jl/src/ACEcore.jl")
using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, test_derivatives
using LinearAlgebra: I
using LinearAlgebra  
# using QuadGK
using SparseArrays: SparseVector, sparse
#using ACE1: rand_radial
using Plots
using Distributions
using Main.ACEcore: SimpleProdBasis
# using ACE1 
using Random
using CSV

@info("Testing Pure2bPIbasis")


"""
This function is for getting all the permutations which we used to calculate the coefficients, currently only for 2b baiss
   param: maxdeg :: Int, max degree
"""
function get_NN(maxdeg)
   index_pairs = Vector{Int64}[[i] for i = 1:maxdeg]
   #lag = 0
   for i = 1:maxdeg
      for j = i:maxdeg
         if (i + j) <= maxdeg && i != j 
            push!(index_pairs, [i ,j])
            #lag = max(i,j)
         end
      end
   end
   return index_pairs
end


"""
Generate cheb_nodes of size N
"""
function chev_nodes(N) # sampling from 0 to 1 for simplicity
   return [cos(j * pi/N) for j=0:N]
end


"""
This function implements the method of getting the P_kappa coeffs for evaluation of pure PI basis, current implementation for 2b PI basis only

param: pibasis :: 1d_pi_basis, target pibasis to be purified
return: P_kappa_coeffs:: Dict{Vector{Int}, SparseVector{Float64, Int64}}(), result of fitting the pibasis for different tuples from get_NN(pibasis)
"""
function P_kappa_prod_coeffs(poly, NN, tol = 1e-10)
   L = 500
   #RR = zeros(L, length(poly))
   sample_points = chev_nodes(L)
   # @show sample_points
   #@show sample_points
   RR = poly(sample_points)
   
   qrF = qr(RR)
   # @show size(RR)
   Pnn = Dict{Vector{Int64}, SparseVector{Float64, Int64}}() # key: the index of correpsonding tuple in the NN list; value: SparseVector
   # @show length(NN)
   # @show NN
   # solve RR * x - Rnn = 0 for each basis
   for nn in NN  #NN contains all the ordered pairs corresponding to the pibasis, pibasis containes all <= N body basis with self-interactions
      Rnn = RR[:, nn[1]]
      #@show size(Rnn)
      #@show nn
      #@show size(qrF)
      for t = 2:length(nn)
         Rnn = Rnn .* RR[:, nn[t]] # do product on the basis according to the tuple nn, would be the ground truth target in the least square problem
      end
      p_nn = map(p -> (abs(p) < tol ? 0.0 : p), qrF \ Rnn) # for each element p in qrF\Rnn, if p is < tol, set it to 0 for converting to sparse matrix
      # @show p_nn
      # @show norm(RR * p_nn - Rnn, Inf)
      @assert norm(RR * p_nn - Rnn, Inf) < tol      
      Pnn[nn] = sparse(p_nn)
      #@show Pnn
   end
   return Pnn
end


"""
This function evaluate basis using different polynomial basis and the corresponding transformation matrix

param: points:: 
return: P_kappa_coeffs:: Dict{Vector{Int}, SparseVector{Float64, Int64}}(), result of fitting the pibasis for different tuples from get_NN(pibasis)
"""
function eval_pure_basis_from_impure(points, poly, NN, C)
   # evaluate all basis at the above M points and summing up
   A_n = sum(poly(points), dims = 1)
   # do the product according to NN specification, basis1(X) will return a Vector with same length as X
   prod_basis_eval = SimpleProdBasis(NN) 
   A_n = vcat(A_n...)
   impure = prod_basis_eval(A_n)
   return C*impure
end


function getG(X1, X2, poly, NN2)
   B = length(NN2) # number of basis
   D1 = poly(X1)
   D2 = poly(X2)
   G = zeros(B, B)
   for i = 1:B
      for j = 1:B
         n, m = NN2[i]
         nn, mm = NN2[j]
         lnx1 = D1[:, n]
         lmx1 = D1[:, m]
         lnx2 = D2[:, n]
         lmx2 = D2[:, m]
         lnnx1 = D1[:, nn]
         lmmx1 = D1[:, mm]
         lnnx2 = D2[:, nn]
         lmmx2 = D2[:, mm]
         G[i, j] = sum((lnx1 .* lmx2' + lmx1 .* lnx2') .* (lnnx1 .* lmmx2' + lmmx1 .* lnnx2')) / length(X1) ^ 2
      end
   end
   return G
end

# -------------------------------------------------------testing coefficients------------------------------------------#
# testing configs
max_degree_list = [15]  # max degree of basis
# Random.seed!(1234)
DOTEST = true
MM = 250
exp_dir = "test_result_Rastrigin_dmaxdegree_samedata/"
distribution = Uniform

Xs, Ys = rand(distribution(-1, 1), MM), rand(distribution(-1, 1), MM)
num_sam_list = [MM]

for max_degree in max_degree_list
   NN = get_NN(max_degree)
   println("Number of basis: ", length(NN))
   println("max_degree: ", max_degree)


   poly = legendre_basis(max_degree, normalize = true) # may have contains basis that are useless
   Pnn_all = P_kappa_prod_coeffs(poly, NN)

   # ==== testing code case to show that the coefficients are correct ====
   if DOTEST == true
      test_index = 2 # test index
      px_list = []
      ground_list = []
      target_nn = NN[test_index]
      coef = Pnn_all[NN[test_index]]
      for tt in LinRange(-1, 1, 100)
         cal = poly(tt)
         ground = length(target_nn) == 2 ? cal[target_nn[1]] * cal[target_nn[2]] : cal[target_nn[1]]
         px = coef.nzval[1] * cal[coef.nzind[1]]

         if length(coef.nzind) >= 2
            for t = 2:length(coef.nzind)
               px += coef.nzval[t] * cal[coef.nzind[t]]
            end
         end
         push!(px_list, px)
         push!(ground_list, ground)
         @assert(norm(px-ground) < 1e-12)
      end
      #print("Done line 134")
      all_truth_value = poly(LinRange(-1, 1, 100))
      truth_value = length(target_nn) == 2 ? all_truth_value[:, target_nn[1]] .* all_truth_value[:, target_nn[2]] : all_truth_value[:, target_nn[1]]
      plot(LinRange(-1, 1, 100), px_list, label = "px_list")
      plot!(LinRange(-1, 1, 100), truth_value, label = "ground")
      # savefig("checkcoef")
   end
   
   # ========

   # -------------------------------------------------------pure basis transformation------------------------------------------#


   S = length(NN)
   C = zeros(S, S) + I(S) # transformation matrix

   # assign coefficent for each corresponding row
   for i = max_degree + 1:S
      pnn = Pnn_all[NN[i]]
      for k = 1:length(pnn.nzind)
         C[i, pnn.nzind[k]] -= pnn.nzval[k]
      end
   end

   if DOTEST == true
      # visualize result

      ground_pure2b = zeros(100, S - max_degree)
      # heatmap(C, color =:grays, aspect_ratio=1)
      # savefig("checkmap")

      # check with theorectical basis on linrange
      X1 = LinRange(-1, 1, 100)
      X2 = LinRange(-1, 1, 100)
      
      see_NN = []
      for i = 1:S - max_degree
         nn, mm = NN[i + max_degree]
         #@show nn, mm
         push!(see_NN, NN[i + max_degree])
         ground_pure2b[:, i] = poly(X1)[:, nn] .* poly(X2)[:, mm] + poly(X1)[:, mm] .* poly(X2)[:, nn]
      end

      # check if the inner product is correct
      for pp = 1:S - max_degree
         for qq = pp : S - max_degree
            ground_pure_1 = ground_pure2b[:, pp]
            ground_pure_2 = ground_pure2b[:, qq]
            ground_pure_12 = zeros(100, 2)
            ground_pure_12[:, 1] = ground_pure_1
            ground_pure_12[:, 2] = ground_pure_2
            d2_int_truth = sum(ground_pure_1 .* ground_pure_2') / (100 ^ 2) # 2d integral approximation
            println("d2_int from ground truth basis :", d2_int_truth)
            if d2_int_truth > 1e-12
               @show see_NN[pp]
               @show see_NN[qq]
            end
         end
      end

      # form the G matrix
      G = zeros(S - max_degree, S - max_degree)
   
      # for pp = 1:S - max_degree
      #    for qq = 1 : S - max_degree
      #       @show pp, qq
      #       @show see_NN[1]
      #       ground_pure_1 = ground_pure2b[:, pp]
      #       ground_pure_2 = ground_pure2b[:, qq]
      #       G[pp, qq] = sum(ground_pure_1 .* ground_pure_2') / (100 ^ 2) # 2d integral approximation
      #       @show G[pp, qq]
      #       if abs(G[pp, qq]) > 1e-12
      #          @show pp, qq
      #       end
      #    end
      # end
      G = getG(X1, X2, poly, NN[max_degree + 1:end])
      @show size(G)
      println("cond(G): ", cond(G))
      pure = zeros(100, S)
      @assert 1 == 0
      println("Check value")


      for k = 1:100
         pure[k, :] = eval_pure_basis_from_impure([X1[k], X2[k]], poly, NN, C)
      end

      pure_12 = zeros(100, 2)

      println(norm(ground_pure_1 - pure[:, findall(x-> x == [n, m], NN)]))   
      println(norm(ground_pure_2 - pure[:, findall(x-> x == [p, q], NN)]))   

      pure_12[:, 1] = pure[:, findall(x-> x == [n, m], NN)]
      pure_12[:, 2] = pure[:, findall(x-> x == [p, q], NN)]
      d2_int_try = sum(pure[:, findall(x-> x == [n, m], NN)] .* pure[:, findall(x-> x == [p, q], NN)]') / 100 ^ 2
      println("d2_int_try from ground truth basis :", d2_int_try)

      @assert d2_int_try < 1e-12
      pi2b_pure = pure[:, max_degree + 1:end]
      
      @show cond(pure_12)
   
   end


   # -------------------------------------------------------test the pure basis and impure basis on fitting problem------------------------------------------#

   # testing sym functions
   function Rastrigin(x, y)
      return sum([x^2 - 10 * cos(2 * pi * x) + 10, y^2 - 10 * cos(2 * pi * y) + 10])
   end

   function trial_func(x, y)
      return x * y + x * y ^ 2 + x^2 * y^2 + x + y
   end

   Testing_func = Rastrigin
   global xs = range(-1, 1, length = 100)
   global ys = range(-1, 1, length = 100)


   mesh_x = xs' .* ones(100)
   mesh_y = ones(100)' .* ys

   ground_zs = map(Testing_func, mesh_x, mesh_y)
   # @show size(ground_zs)
   surface(xs, ys, ground_zs)
   savefig("test/testingsurface")

   # we solve the problem with least square fitting with L2 regularization

   pure_err = []
   impure_err = []
   lina = 200
   linb = 300
   # num_sam_list = LinRange(lina, linb, 21)
   cond_num_pure = []
   cond_num_impure = []
   ratio_cond = []

   for num_sam in num_sam_list
      num_sam = Integer(num_sam)
      println("----------------------------------------------")
      println("current number of sample points: ", num_sam)
      path = exp_dir * "maxdeg="*string(max_degree)*"/num_sam="*string(num_sam)*"/"
      mkpath(path)
      # draw sample from uniform distribution
      Xs, Ys = range(-1, 1, length = num_sam), range(-1, 1, length = num_sam)


      A_pure = zeros(num_sam, length(NN))
      A_impure = zeros(num_sam, length(NN))

      B = zeros(num_sam)
      #@time begin

      # pure_for_map_eval(pp::Vector{Float64}) = eval_pure_basis_from_impure(pp, poly, NN, C)
      # impure_for_map_eval(pp::Vector{Float64}) = eval_pure_basis_from_impure(pp, poly, NN, zeros(S, S) + I(S))
      # B = map(Testing_func, Xs, Ys)
      # A_pure = map(pure_for_map_eval, Xs_map)
      # A_impure = map(impure_for_map_eval, Xs_map)

      
      for k = 1:num_sam
         B[k] = Testing_func(Xs[k],Ys[k])
         A_pure[k, :] = eval_pure_basis_from_impure([Xs[k], Ys[k]], poly, NN, C)
         A_impure[k, :] = eval_pure_basis_from_impure([Xs[k], Ys[k]], poly, NN, zeros(S, S) + I(S))
      end
      

      @show cond(A_impure)
      @show cond(A_pure)
      
      push!(cond_num_impure, cond(A_impure))
      push!(cond_num_pure, cond(A_pure))
      push!(ratio_cond, cond(A_impure)/cond(A_pure))
      
      # x = (λI + A'A) * A' * B
      
      # # solve the problem with inv
      # λ = 0.1
      # sol_pure = inv(λ * I(length(NN)) + transpose(A_pure) * A_pure) * transpose(A_pure) * B
      # sol_impure = inv(λ * I(length(NN)) +transpose(A_impure) * A_impure) * transpose(A_impure) * B
      
      # solve the problem with qr
      LL = length(NN)
      λ = 0.1
      sol_pure = qr(vcat(A_pure, λ * I(LL) + zeros(LL, LL))) \ vcat(B, zeros(LL))
      sol_impure = qr(vcat(A_impure, λ * I(LL) + zeros(LL, LL))) \ vcat(B, zeros(LL))

      # solve the problem with ARD
      
      # ARD = pyimport("sklearn.linear_model")["ARDRegression"]
      # clf = ARD()
      # sol_pure = clf.fit(A_pure, B).coef_
      # sol_impure = clf.fit(A_impure, B).coef_
      

      
      # try fitting with linspace for plotting
      xs = ys = range(-1,1, length=100)
      zs_pure = [dot(sol_pure, eval_pure_basis_from_impure([x, y], poly, NN, C)) for y in ys, x in xs]
      zs_impure = [dot(sol_impure, eval_pure_basis_from_impure([x, y], poly, NN, zeros(S, S) + I(S))) for y in ys, x in xs]

      surface(xs, ys, zs_pure)
      savefig(path*"purefittingsurface" * string(num_sam))
      surface(xs, ys, zs_impure)
      savefig(path*"impurefittingsurface" * string(num_sam))
      println("relative error of pure basis: ", norm(zs_pure - ground_zs)/norm(ground_zs))
      println("relative error of Impure basis: ", norm(zs_impure - ground_zs)/norm(ground_zs))
      println("is cond(A_pure) < cond(A_impure)? :", cond(A_pure) < cond(A_impure))
      push!(pure_err, norm(zs_pure - ground_zs)/norm(ground_zs))
      push!(impure_err, norm(zs_impure - ground_zs)/norm(ground_zs))

   end
   path = exp_dir * "maxdeg="*string(max_degree)
   mkpath(path)

   p1 = plot(num_sam_list, pure_err, label = "pure", xlabel = "number of sampling points", ylabel = "rel err", yscale = :log10)
   plot!(num_sam_list, impure_err, label = "impure", yscale = :log10)
   savefig(p1, path*"/checkerror"*string(lina)*"_"*string(linb))

   p2 = plot(num_sam_list, cond_num_pure, label = "pure", xlabel = "number of sampling points", ylabel = "Coniditional number")
   plot!(num_sam_list, cond_num_impure, label = "impure")
   savefig(p2, path*"/checkcond"*string(lina)*"_"*string(linb))

   CSV.write(path*"/err_"*string(lina)*"_"*string(linb)*".csv", (pure_err = pure_err, impure_err = impure_err)) 
   CSV.write(path*"/cond_"*string(lina)*"_"*string(linb)*".csv", (cond_num_pure = cond_num_pure, cond_num_impure = cond_num_impure, ratio_cond = ratio_cond)) 


   @show ratio_cond
end
































