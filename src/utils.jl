using Polynomials4ML, Test
using Polynomials4ML: evaluate, evaluate_d, evaluate_dd
using Polynomials4ML.Testing: println_slim, test_derivatives
using LinearAlgebra: I
using LinearAlgebra  
using SparseArrays: SparseVector, sparse
using Distributions
# using Main.ACEcore: SimpleProdBasis
using Random

"""
This function is for getting all the permutations which we used to calculate the coefficients, currently only for 2b baiss
   param: maxdeg :: Int, max degree
"""
function get_NN(maxdeg)
   index_pairs = Vector{Int64}[[i] for i = 1:maxdeg]
   for i = 1:maxdeg
      for j = i:maxdeg
         if (i + j) <= maxdeg
            push!(index_pairs, [i ,j])
         end
      end
   end
   return index_pairs
end


"""
This function return the G matrix consisting of the inner prodcut of basis

param: X1: points of first dimension
param: X2: points of second dimension
param: NN2: a list of list with length = 2, sbould be of length = number of basis
param: poly: type of polynomial basis to be used
return: G
"""
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


"""
This function implements the method of getting the P_kappa coeffs for evaluation of pure PI basis, current implementation for 2b PI basis only

param: pibasis :: 1d_pi_basis, target pibasis to be purified
return: P_kappa_coeffs:: Dict{Vector{Int}, SparseVector{Float64, Int64}}(), result of fitting the pibasis for different tuples from get_NN(pibasis)
"""
function P_kappa_prod_coeffs(poly, NN, tol = 1e-10)
   L = 500
   #RR = zeros(L, length(poly))
   # sample_points = chev_nodes(L)

   sample_points = rand(Uniform(-1, 1), L)
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

