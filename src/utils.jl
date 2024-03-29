using Polynomials4ML, Test
using Polynomials4ML.Testing: println_slim, test_derivatives 
using SparseArrays: SparseVector, sparse
# using Main.ACEcore: SimpleProdBasis
using LinearAlgebra, Random, Plots, CSV, StaticArrays, Distributions
using JuLIP, StaticArrays, PyCall

"""
This function is for getting all the permutations which we used to calculate the coefficients, currently only for 2b baiss
   param: maxdeg :: Int, max degree
"""
function get_NN(maxdeg, correlation_order)
   index_pairs = Vector{Int64}[[i] for i = 1:maxdeg]
   #lag = 0
   if correlation_order >= 2
      for i = 1:maxdeg
         for j = i:maxdeg
            if (i + j) <= maxdeg
               push!(index_pairs, [i ,j])
            end
         end
      end
   end

   if correlation_order >= 3
      for i = 1:maxdeg
         for j = i:maxdeg
            for k = j: maxdeg
               if (i + j + k) <= maxdeg
                  push!(index_pairs, [i ,j, k])
               end
            end
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
function getIntegralG(X1, X2, poly, NN2)
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

"""
This function implements the method of getting the P_kappa coeffs for evaluation of pure PI basis, current implementation for 2b PI basis only

param: pibasis :: 1d_pi_basis, target pibasis to be purified
return: P_kappa_coeffs:: Dict{Vector{Int}, SparseVector{Float64, Int64}}(), result of fitting the pibasis for different tuples from get_NN(pibasis)
"""
function rand_radial(poly, N)
   if typeof(poly) == typeof(legendre_basis)
      return rand(Uniform(-1, 1), N)
   elseif  typeof(poly) == typeof(chebyshev_basis)
      return "to be implementated"
   else
      println("Implementation Not found")
   end
end

"""
    designMatNB(train, poly_basis, max_deg, ord; body=:TwoBodyThreeBody)
    Returns the design matrix for averaged estimation 1D model
    
    * @param train          Vector{Vector{Vector(Float64)}}     set of atomic strucutres (1d) is array of array of radius distances
    * @param poly_basis     OrthPolyBasis1D3T{Float64}          basis function
    * @param max_deg        Int64                               maximum degree of poly basis function
    * @param body           Int64                               body order
    * @return               Matrix{ComplexF64}                  design matrix

    # Examples
    ```jldoctest
    num_sam = 10
    K_R = 4
    train = rand(num_sam, K_R)
    poly_basis = chebyshev_basis(10)
    max_deg = 5
    ord = 2
    A = designMatNB2D(train, poly_basis, max_deg, ord; body=:TwoBodyThreeBody)
    ```
"""
function designMatNB(train, poly_basis, max_deg, ord; body=:TwoBodyThreeBody)
   NN = get_NN(max_deg, ord)
   NN2b = NN[length.(NN) .== 1]
   NN3b = NN[length.(NN) .== 2]
   M, K_R = size(train)

   poly_list = [poly_basis(train[:, i]) for i = 1:K_R]
   if body == :TwoBody # 2body interaction
       A = zeros(M, length(NN2b))
       for i = 1:length(NN2b)
           nn = NN2b[i]
           A[:, i] = sum([PX1[:, nn] for PX1 in poly_list])
       end
   elseif body == :ThreeBody #3body interaction
       A = zeros(M, length(NN3b))
       for i = 1:length(NN3b)
           nn, mm = NN3b[i]
           A[:, i] = sum([PX1[:, nn] .* PX2[:, mm] for PX1 in poly_list for PX2 in poly_list if PX1 != PX2])
       end
   elseif body == :TwoBodyThreeBody #both 2b3b
       A = zeros(M, length(NN))
       for i = 1:length(NN2b)
           nn = NN2b[i]
           A[:, i] = sum([PX1[:, nn] for PX1 in poly_list])
       end
       for i = 1:length(NN3b)
           nn, mm = NN3b[i]
           A[:, length(NN2b) + i] = sum([PX1[:, nn] .* PX2[:, mm] for PX1 in poly_list for PX2 in poly_list if PX1 != PX2])
       end
   else
       println("Does not support this body order.")
   end
   return A / sqrt(K_R) # assume if we are fitting ord-body energy, we use the ord which makes sense
end

"""
    predMatNB(test, poly_basis, max_deg, ord; body=:TwoBodyThreeBody)
    Returns the prediction basis matrix for _body_ interaction pair potential term
    
    * @param test           Vector{Vector{Vector(Float64)}}     set of atomic strucutres (1d) is array of array of radius distances pair tuple
    * @param poly_basis     OrthPolyBasis1D3T{Float64}          basis function
    * @param max_deg        Int64                               maximum degree of poly basis function
    * @param body           Int64                               body order
    * @return               Matrix{ComplexF64}                  basis matrix

    # Examples
    ```jldoctest
    testSampleSize = 10
    K_R_test = 2
    DD_test = [rand(distribution(-1, 1), K_R_test) for _=1:testSampleSize]
    XX_test = reduce(vcat, DD_test')
    DD_test_pair = permDist(DD_test, ord)
    DD_test = reduce(vcat, reduce(vcat, DD_test_pair'))
    A_test = predMatNB(DD_test, poly, max_degree, ord; body = body_order)
    ```
"""
# TODO: Fix K_R>2 issue
function predMatNB(test, poly_basis, max_deg, ord; body=:TwoBodyThreeBody)
    NN = get_NN(max_deg, ord)
    NN2b = NN[length.(NN) .== 1]
    NN3b = NN[length.(NN) .== 2]

    if length(size(test)) == 1
        M = length(test)
        K_R = 1
    else
        M, K_R = size(test)
    end
 
    poly_list = [poly_basis(test[:, i]) for i = 1:K_R]
    if body == :TwoBody # 2body interaction
        return poly_basis(test)
    elseif body == :ThreeBody #3body interaction
        @assert(K_R == 2)
        poly_list = [poly_basis(test[:, i]) for i = 1:K_R]
        A_test = zeros(M, length(NN3b))
        for i = eachindex(NN3b)
            nn, mm = NN3b[i]
            # @show(size(poly_list[1][:, nn]' .* poly_list[2][:, mm]'))
            A_test[:, i] = poly_list[1][:, nn] .* poly_list[2][:, mm] + poly_list[1][:, mm] .* poly_list[2][:, nn]
            # A_test[:, i] = sum([PX1[:, nn] .* PX2[:, mm] for PX1 in poly_list for PX2 in poly_list if PX1 != PX2])
        end
        # A_test = zeros(M, length(NN3b))
        # for i = eachindex(NN3b)
        #     nn, mm = NN3b[i]
        #     A_test[:, i] = sum([PX1[:, nn] .* PX2[:, mm] for PX1 in poly_list for PX2 in poly_list if PX1 != PX2])
        # end
    elseif body == :TwoBodyThreeBody #both 2b3b
        A_test = zeros(M, length(NN))
        A_test[:, 1:length(NN2b)] = poly_basis.(test)
        poly_list = [poly_basis(test[:, i]) for i = 1:2]
        A_test = zeros(M, length(NN3b))
        for i = eachindex(NN3b)
            nn, mm = NN3b[i]
            A_test[:, i] = sum([poly_list[1][:, nn] .* poly_list[2][:, mm]])
        end
    else
        println("Does not support this body order.")
    end
    return A_test
 end

"""
    solveLSQ(A_pure, Y; λ=0.1, solver=:qr)
    Returns the paraemter solution of a linear least square problem

    * @param A_pure         Matrix{ComplexF64}      design matrix
    * @param Y              Vector{ComplexF64}      target value
    * @keywordparam λ       Int64                   L2 regularizer coefficient
    * @keywordparam solver  :qr or :ard             QR solver or ARD solver
    * @return               Vector{ComplexF64}      parameters θ
"""
function solveLSQ(A_pure, Y; λ=0.1, solver=:qr)
   if solver == :qr
      # solve the problem with qr
      LL = size(A_pure)[2]
      sol_pure = qr(vcat(A_pure, λ * I(LL) + zeros(LL, LL))) \ vcat(Y, zeros(LL))
   elseif solver == :ard
   # solve the problem with ARD       
      ARD = pyimport("sklearn.linear_model")["ARDRegression"]
      clf = ARD(fit_intercept=false).fit(A_pure, Y)
      sol_pure = clf.coef_
   end
   return sol_pure
end


"""
This function generates positions of atoms as a vector in R^dim given a data distribution

param: dst :: Int, data distribution, usually uniform
param: dim :: Int, dimension of the atomic environment, 1D, 2D or 3D
param: num_of_atoms :: Int, number of atoms in the atmoic configuration
return: result :: Vector{Vector{Float64}}, an array of atom positions
"""
function gen_correlated_pos(dst, dim, num_of_atoms)
    result = [rand(dst, dim) for _=1:num_of_atoms]
    return result
end

"""
This function generates positions of atoms that is more realistic

keyparam: spc :: speices of atoms
return: at, r_nn, X, Eref :: atomic environment, nearest radius distance, positions of atomis and reference energy
"""
function atom_bulk(spc=:Al)
    at = bulk(spc, cubic=true) * 3
    rattle!(at, 0.03)
    r_nn = rnn(:W)
    X = copy(positions(at))
    
    calc = StillingerWeber()
    Eref = energy(calc, at)
#     F = forces(pB, at)
    
    return at, r_nn, X, Eref
end

"""
This function converts positions of atoms to radius distance vectors based on body interaction order

param: pos :: Vector{Vector{Float64}}, positions of atoms
param: order :: Int{2, 3}, body interaction order
return: dis :: Vector{Vector{Float64}}, radius distance vector between #order body atomic interaction
"""
function pos_to_dist(pos, order)
    if order == 2
        dis = zeros(Float64, binomial(length(pos), 2))
        d=1
        for i=eachindex(pos)
            for j=i:length(pos)
                if i != j
                    dis[d] = norm(pos[i] - pos[j])
                    d += 1
                end
            end
        end
        @assert(length(dis) == d-1)
        return dis
   #  elseif order == 3
   #    @show binomial(length(pos), 3)
   #      dis = zeros(Float64, binomial(length(pos), 3))
   #      @show length(dis)
   #      d=1
   #      for i=eachindex(pos)
   #          for j=i:length(pos)
   #             for k=j:length(pos)
   #                if i != j && j != k
   #                   @show d
   #                   dis[d] = norm(pos[i] - pos[j])
   #                   dis[d+1] = norm(pos[i] - pos[k])
   #                   d += 2
   #                end
   #             end
   #          end
   #      end
   #      @assert(length(dis) == d-1)
   #      return dis
    end
end

"""
This function converts an array of radius distances to ord pair radius distance vectors based on body interaction order

param: D :: Vector{Vector{Float64}}, array of raidus distances
param: ord :: Int{2, 3}, body interaction order
return: DD :: Vector{Vector{Float64}}, radius distance vector between #order body atomic interaction
"""
function permDist(D, ord)
    DD = Vector[]
    if ord == 1
        return D
    elseif ord == 2
        for m = eachindex(D)
            R = D[m]
            R_pair = Vector[]
            pair_index = 0
            stop = binomial(length(R), ord)
            for i = eachindex(R)
                for j = eachindex(R)
                    if i != j && pair_index != stop
                        push!(R_pair, [R[i], R[j]])
                        pair_index += 1
                    end
                end
            end
            @assert(length(R_pair) == stop)
            push!(DD, R_pair)
        end
    else
        println("Does not support this body order.")
    end
    return DD
 end

# ----- 2D implementation ------
"""
    spatial2rad(xs)
    Map (x, y) → (r, θ) in 2D

    * @param xs          Vector{Vector(Float64)}                  angular info of xs
    * @return xs_re      Vector{Vector(Float64)}                  exp basis matrix
"""
function spatial2rad(xs)
    num_sam, K_R, _ = size(xs)
    xs_re = zeros(size(xs))
    for i = 1:num_sam        
        for j = 1:K_R
            xs_re[i,j,1], xs_re[i,j,2] = sqrt(xs[i, j, 1]^2 + xs[i, j, 2]^2), atan(xs[i, j, 2]/xs[i, j, 1])
        end
    end
    return xs_re
end

"""
    exp_basis(xs_theta, deg)
    Returns the basis matrix for expotential basis
    
    * @param xs_theta          Vector{Vector(Float64)}          angular info of xs
    * @param deg               Int                              max deg of exp function
    * @return B             Matrix{ComplexF64}                  exp basis matrix
"""
function exp_basis(xs_theta, deg)
    num_sam = length(xs_theta)
    B = zeros(ComplexF64, num_sam, 2 * deg + 1)
    for k = 1:num_sam
        for j = 1:2*deg + 1
            pow = j - deg - 1
            B[k, j] = exp(1im * pow * 2 * pi * xs_theta[k])
        end
    end
    return B
end


function get_NN_exp(max_deg_exp, rotationInv = true)
    if rotationInv == true
        NN_exp2b = [[0]]
        NN_exp3b = [[i, -i] for i = 0:max_deg_exp]
        return NN_exp2b, NN_exp3b
   else
        NN_exp2b = [[0]]
        NN_exp3b = [[i, -j] for i = 0:max_deg_exp for j = 0:max_deg_exp]
        return NN_exp2b, NN_exp3b
   end
end

# function dist_transform_2d(Xs)
    


"""
    designMatNB2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody)
    Returns the design matrix for averaged estimation 2D model and spec
    
    * @param train          Vector{Vector{Vector(Float64)}}     set of atomic strucutres (1d) is array of array of radius distances
    * @param poly_basis     OrthPolyBasis1D3T{Float64}          basis function
    * @param max_deg_poly   Int64                               maximum degree of poly basis function
    * @param max_deg_exp    Int64                               maximum degree of exp basis function
    * @param body           Int64                               body order
    * @return               Matrix{ComplexF64}                  design matrix
    * @return               Vector{tuple(Int)}                  specification of the design matrix

    # Examples
    ```jldoctest
    num_sam = 10
    K_R = 4
    train = rand(num_sam, K_R, 2)
    poly_basis = chebyshev_basis(10)
    max_deg_poly = 5
    max_deg_exp = 5
    ord = 2
    A, spec = designMatNB2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody)
    ```
"""
function designMatNB2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody, rotationInv = true) # TODO: remove ord later
   if body == :TwoBody
        ord = 1
   else
        ord = 2
   end

   NN = get_NN(max_deg_poly, ord)
   NN2b = NN[length.(NN) .== 1]
   NN3b = NN[length.(NN) .== 2]
   NN_exp2b, NN_exp3b = get_NN_exp(max_deg_exp, rotationInv)
   M, K_R, _ = size(train)
   xs_rad = spatial2rad(train) # num_data × K_R × 2
   @show [maximum(xs_rad[:,i,1]) for i = 1:K_R]
   @show maximum(poly_basis(xs_rad[:,1,1]))
   poly_list = [poly_basis(xs_rad[:, i, 1]) for i = 1:K_R] #  K_R × num_data × length(poly_basis) 
   exp_list = [exp_basis(xs_rad[:, i, 2], max_deg_exp) for i = 1:K_R] #  K_R × num_data × max_deg_exp
   spec = []
   @show maximum(poly_list[1])
   
   if body == :TwoBody # 2body interaction
        A = zeros(ComplexF64, M, length(NN2b) * length(NN_exp2b))
       for i = 1:length(NN2b)
           nn = NN2b[i]
           for j = 1:length(NN_exp2b)
               #@show i, j
               #@show (i-1) * length(NN_exp2b) + j
               pow = NN_exp2b[j] # actual power
               pp = pow .+ max_deg_exp .+ 1 # index in the exp_list
               A[:, (i-1) * length(NN_exp2b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, pp] for p = 1:K_R ])
               push!(spec, [(nn[1], pp[1])])
           end
       end
   elseif body == :ThreeBody #3body interaction
       A = zeros(ComplexF64, M, length(NN3b) * length(NN_exp3b))
       for i = 1:length(NN3b)
           nn, mm = NN3b[i]
           for j = 1:length(NN_exp3b)
               pow_pp, pow_qq = NN_exp3b[j]
               pp, qq = pow_pp .+ max_deg_exp .+ 1, pow_qq .+ max_deg_exp .+ 1 # index in the exp_list
               A[:, (i-1) * length(NN_exp3b) + j] = sum([poly_list[p][:, nn] .* poly_list[q][:, mm] .* exp_list[p][:, pp] .* exp_list[q][:, qq] for p=1:K_R for q=1:K_R if (poly_list[p] != poly_list[q] || exp_list[p] != exp_list[q])])
               push!(spec, [(nn, pp), (mm, qq)])
           end
       end
   elseif body == :TwoBodyThreeBody #both 2b3b
       A = zeros(ComplexF64, M, length(NN2b) * length(NN_exp2b) + length(NN3b) * length(NN_exp3b))
       for i = 1:length(NN2b)
        nn = NN2b[i]
        for j = 1:length(NN_exp2b)
            pow = NN_exp2b[j] # actual power
            pp = pow .+ max_deg_exp .+ 1 # index in the exp_list
            A[:, (i-1) * length(NN_exp2b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, pp] for p = 1:K_R ])
            push!(spec, [(nn[1], pp[1])])
        end
       end
       for i = 1:length(NN3b)
        nn, mm = NN3b[i]
        for j = 1:length(NN_exp3b)
            pow_pp, pow_qq = NN_exp3b[j]
            pp, qq = pow_pp .+ max_deg_exp .+ 1, pow_qq .+ max_deg_exp .+ 1 # index in the exp_list
            A[:, length(NN2b) * length(NN_exp2b) + (i-1) * length(NN_exp3b) + j] = sum([poly_list[p][:, nn] .* poly_list[q][:, mm] .* exp_list[p][:, pp] .* exp_list[q][:, qq] for p=1:K_R for q=1:K_R if (poly_list[p] != poly_list[q] || exp_list[p] != exp_list[q])])
            push!(spec, [(nn, pp), (mm, qq)])
        end
    end
   else
       println("Does not support this body order.")
   end
   @show maximum(real(A))
   return real(A) / sqrt(K_R), spec
end


"""
    designMat3B2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody)
    Returns the design matrix for averaged estimation 2D model and spec, assuming no-correlation between data input vectors.
    
    * @param train          Vector{Vector{Vector(Float64)}}     set of atomic strucutres (1d) is array of array of radius distances
    * @param poly_basis     OrthPolyBasis1D3T{Float64}          basis function
    * @param max_deg_poly   Int64                               maximum degree of poly basis function
    * @param max_deg_exp    Int64                               maximum degree of exp basis function
    * @param body           Int64                               body order
    * @return               Matrix{ComplexF64}                  design matrix
    * @return               Vector{tuple(Int)}                  specification of the design matrix

    # Examples
    ```jldoctest
    num_sam = 10
    K_3 = 4
    train = rand(num_sam, K_3, 2)
    poly_basis = chebyshev_basis(10)
    max_deg_poly = 5
    max_deg_exp = 5
    ord = 2
    A, spec = designMatNB2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody)
    ```
"""
function designMat3B2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody, rotationInv = true) # TODO: remove ord later
   if body == :TwoBody
        ord = 1
   else
        ord = 2
   end
   
#     if body == :TwoBody # 2body interaction
#         A = zeros(ComplexF64, M, length(NN2b) * length(NN_exp2b))
#         for i = 1:length(NN2b)
#             nn = NN2b[i]
#             for j = 1:length(NN_exp2b)
#                 #@show i, j
#                 #@show (i-1) * length(NN_exp2b) + j
#                 pow = NN_exp2b[j] # actual power
#                 pp = pow .+ max_deg_exp .+ 1 # index in the exp_list
#                 A[:, (i-1) * length(NN_exp2b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, pp] for p = 1:K_N ])
#                 push!(spec, [(nn[1], pp[1])])
#             end
#         end
#     elseif body == :ThreeBody #3body interaction
#         A = zeros(ComplexF64, M, length(NN3b) * length(NN_exp3b))
#         for i = 1:length(NN3b)
#             nn, mm = NN3b[i]
#             for j = 1:length(NN_exp3b)
#                 pow_pp, pow_qq = NN_exp3b[j]
#                 pp, qq = pow_pp .+ max_deg_exp .+ 1, pow_qq .+ max_deg_exp .+ 1 # index in the exp_list
#                 A[:, (i-1) * length(NN_exp3b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, mm] for p=1:K_N ])
#                 push!(spec, [(nn, pp), (mm, qq)])
#             end
#         end
#     elseif body == :TwoBodyThreeBody #both 2b3b
#         A = zeros(ComplexF64, M, length(NN2b) * length(NN_exp2b) + length(NN3b) * length(NN_exp3b))
#         for i = 1:length(NN2b)
#             nn = NN2b[i]
#             for j = 1:length(NN_exp2b)
#                 pow = NN_exp2b[j] # actual power
#                 pp = pow .+ max_deg_exp .+ 1 # index in the exp_list
#                 A[:, (i-1) * length(NN_exp2b) + j] = sum([poly_list[p][:, nn] for p=1:K_N ])
#                 push!(spec, [(nn[1], pp[1])])
#             end
#         end
#         for i = 1:length(NN3b)
#             nn, mm = NN3b[i]
#             for j = 1:length(NN_exp3b)
#                 pow_pp, pow_qq = NN_exp3b[j]
#                 pp, qq = pow_pp .+ max_deg_exp .+ 1, pow_qq .+ max_deg_exp .+ 1 # index in the exp_list
#                 A[:, length(NN2b) * length(NN_exp2b) + (i-1) * length(NN_exp3b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, pp] for p=1:K_N])
#                 push!(spec, [(nn, pp), (mm, qq)])
#             end
#         end
#     else
#        println("Does not support this body order.")
#     end
#     @show maximum(real(A))
    NN = get_NN(max_deg_poly, ord)
    NN2b = NN[length.(NN) .== 1]
    NN3b = NN[length.(NN) .== 2]
    # NN_exp2b, NN_exp3b = get_NN_exp(max_deg_exp, rotationInv)
    NN_exp2b, NN_exp3b = [[0]], [NN3b]
    @show NN_exp3b

    M, K_N, _ = size(train)
    
    xs_rad = spatial2rad(train) # num_data × K_N × 2
    @show [maximum(xs_rad[:,i,1]) for i = 1:K_N]
    @show maximum(poly_basis(xs_rad[:,1,1]))
    poly_list = [poly_basis(xs_rad[:, i, 1]) for i = 1:K_N] #  K_N × num_data × length(poly_basis) 
    exp_list = [exp_basis(xs_rad[:, i, 2], max_deg_exp) for i = 1:K_N] #  K_N × num_data × max_deg_exp
    spec = []
    @show maximum(poly_list[1])

    if body == :TwoBody # 2body interaction
        A = zeros(ComplexF64, M, length(NN2b) * length(NN_exp2b))
        for i = 1:length(NN2b)
            nn = NN2b[i]
            for j = 1:length(NN_exp2b)
                #@show i, j
                #@show (i-1) * length(NN_exp2b) + j
                pow = NN_exp2b[j] # actual power
                pp = pow .+ max_deg_exp .+ 1 # index in the exp_list
                A[:, (i-1) * length(NN_exp2b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, pp] for p = 1:K_N ])
                push!(spec, [(nn[1], pp[1])])
            end
        end
    elseif body == :ThreeBody #3body interaction
        A = zeros(ComplexF64, M, length(NN3b) * length(NN_exp3b))
        for i = 1:length(NN3b)
            nn, mm = NN3b[i]
            for j = 1:length(NN_exp3b)
                # pow_pp, pow_qq = NN_exp3b[j]
                # pp, qq = pow_pp .+ max_deg_exp .+ 1, pow_qq .+ max_deg_exp .+ 1 # index in the exp_list
                # pow_pp = NN_exp3b[j][2]
                # @show(pow_pp)
                # pp = pow_pp .+ max_deg_exp .+ 1
                # @show(pp)
                pp = mm + max_deg_exp + 1
                A[:, (i-1) * length(NN_exp3b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, pp] for p=1:K_N ])
                # poly_list[p][:, nn] .* poly_list[q][:, mm] .* exp_list[p][:, pp] .* exp_list[q][:, qq] for p=1:K_R for q=1:K_R
                push!(spec, [(nn, pp)])
            end
        end
    elseif body == :TwoBodyThreeBody #both 2b3b
        A = zeros(ComplexF64, M, length(NN2b) * length(NN_exp2b) + length(NN3b) * length(NN_exp3b))
        for i = 1:length(NN2b)
            nn = NN2b[i]
            for j = 1:length(NN_exp2b)
                pow = NN_exp2b[j] # actual power
                pp = pow .+ max_deg_exp .+ 1 # index in the exp_list
                A[:, (i-1) * length(NN_exp2b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, pp] for p=1:K_N ])
                push!(spec, [(nn[1], pp[1])])
            end
        end
        for i = 1:length(NN3b)
            nn, mm = NN3b[i]
            for j = 1:length(NN_exp3b)
                pow_mm = NN_exp3b[j][2]
                mm = pow_mm .+ max_deg_exp .+ 1 # index in the exp_list
                A[:, length(NN2b) * length(NN_exp2b) + (i-1) * length(NN_exp3b) + j] = sum([poly_list[p][:, nn] .* exp_list[p][:, mm] for p=1:K_N ])
                push!(spec, [(nn, mm)])
            end
        end
    else
        println("Does not support this body order.")
    end
    A = real(A)
    @show(maximum(A))
    return A / sqrt(K_N), spec
end