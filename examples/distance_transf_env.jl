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
ϕ(r) = r^(-12) - 2*r^(-6)

# morse transform
x2(r) = 1 / (1 + r / r_nn)

#Agnesi Transform
# 0.33 = a = (p-1)/(p+1)
x1(r) = 1.0 / (1.0 + 0.33*(r / r_nn)^2)

x_in = x1(0.0); x_cut = x1(r_cut) # Regularize til r=0
x(r, x_transf) = 2 * (x_transf(r) - x_cut) / (x_in - x_cut) - 1

# envelope
env(r) = (r^(-p) - r_cut^(-p) + p * r_cut^(-p-1) * (r - r_cut)) * (r < r_cut)

E_avg(X, f) = mean([f.(X[:, i]) for i = 1:size(X)[2]]) * sqrt(size(X)[2])

f = ϕ

distribution=Uniform

domain_lower=r_in
domain_upper=r_cut+1

Xs = range(domain_lower, r_cut, 1000)
# Rs = range(domain_lower, domain_upper, 1000)
Agnesi_transf_Xs = x.(Xs, x1)
Morse_transf_Xs = x.(Xs, x2)
env_Agn_Rs = env.(Agnesi_transf_Xs)
env_Mor_Rs = env.(Morse_transf_Xs)


#save plot
exp_dir = "results/" # result saving dir
mkpath(exp_dir)

let
    fig, axs = PyPlot.subplots(1,3, figsize=(10, 2))
    # sharex=true, sharey = true,

    axs[1][:plot](Xs, ϕ.(Xs), label=L"V_1")
    axs[1].legend(loc="upper right", fontsize = "x-small")
    axs[2][:plot](Xs, Agnesi_transf_Xs, label="Agnesi")
    axs[2][:plot](Xs, Morse_transf_Xs, label="Morse", linestyle="dashed")
    axs[2].legend(loc="upper right", fontsize = "x-small")
    axs[3][:plot](Xs, ϕ.(env_Agn_Rs), label="env with Agnesi")
    axs[3][:plot](Xs, ϕ.(env_Mor_Rs), label="env with Morse", linestyle="dashed")
    axs[3].legend(loc="lower left", fontsize = "x-small")

    fig.savefig(exp_dir*"distance_transf_env", bbox_inches="tight")
    fig
end
