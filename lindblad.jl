### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 104b6690-212b-11ec-3a2b-25c420f62534
begin
	using Plots
	using Plots.PlotMeasures

	using LinearAlgebra
	using Kronecker
	using PlutoUI
	using SparseArrays
	using FiniteDifferences
	#using PlutoTest
	#https://juliapackages.com/p/kronecker
	md"""## Packages"""
end

# ╔═╡ 21712488-f636-4861-913c-012693cb524d
md"""# Order of tensor product basis (TPB)"""

# ╔═╡ 89b58dd0-c70c-4566-b991-2d739990f2ff
md"""
Note, that throughout the thesis a colexicographical ordering of the tensor product is preferred:
$\ket{a_1 b_1}, \ket{a_2 b_1}, \ket{a_1 b_2}, \ket{a_2 b_2}$. In this case the tensor product is realized via the conventional Kronecker product but using flipped arguments:
$A \otimes B \rightarrow \boldsymbol{B} \otimes \boldsymbol{A}$.

Here you can choose the ordering of the tensor product basis of the tensor product ``\mathcal{H}_{\mathcal{A}} \otimes \mathcal{H}_{\mathcal{B}}``:

👉 $(@bind order Select(["colex" => "Colexicographical order", "lex" => "Lexicographical order"]))
"""

# ╔═╡ 23c019dd-53d2-461e-a6ba-f1b5da2f6710
if order == "colex"
md"""
``\begin{pmatrix}{\color{magenta}a}&{\color{red}C}\\ \boldsymbol{b} & D   \end{pmatrix} \overset{\text{c}} {\otimes}\begin{pmatrix} 1 & \large 3 \\ {\color{blue}2} & \scriptsize {\color{green}4}\end{pmatrix} = \begin{pmatrix}{\color{magenta}a}1 & {\color{red}C}1 & {\color{magenta}a} \large 3 & {\color{red}C} \large 3 \\\boldsymbol{b}1 & D1 & \boldsymbol{b} \large 3 & D \large 3 \\ {\color{magenta}a}{\color{blue}2} & {\color{red}C}{\color{blue}2} & {\color{magenta}a}\scriptsize {\color{green}4} & {\color{red}C}\scriptsize {\color{green}4} \\ \boldsymbol{b}{\color{blue}2} & D{\color{blue}2} & \boldsymbol{b}\scriptsize {\color{green}4} & D\scriptsize {\color{green}4} \end{pmatrix}, \quad A \overset{\text{c}}{\otimes} B:= \textbf{B} \otimes \textbf{A}``
"""
else
md"""
``\begin{pmatrix}{\color{magenta}a}&{\color{red}C}\\ \boldsymbol{b} & D   \end{pmatrix} {\otimes}\begin{pmatrix} 1 & \large 3 \\ {\color{blue}2} & \scriptsize {\color{green}4}\end{pmatrix}  = \begin{pmatrix}{\color{magenta}a}1 & {\color{magenta}a} \large 3 & {\color{red}C}1 & {\color{red}C} \large 3 \\{\color{magenta}a}{\color{blue}2} & {\color{magenta}a}\scriptsize {\color{green}4} & {\color{red}C}{\color{blue}2} & {\color{red}C}\scriptsize {\color{green}4} \\  \boldsymbol{b}1 & \boldsymbol{b} \large 3 & D1 & D \large 3 \\  \boldsymbol{b}{\color{blue}2} & \boldsymbol{b}\scriptsize {\color{green}4} & D{\color{blue}2} & D\scriptsize {\color{green}4} \end{pmatrix}``
"""
end

# ╔═╡ f5992b42-ff99-479a-b093-70a9bfd85655
@show order

# ╔═╡ 0a95d032-8576-492f-bee1-ef67303bcb3a
begin 
	if order == "colex"
		⊗(A,B) = kron(B,A)
	else
		⊗(A,B) = kron(A,B)
	end
end

# ╔═╡ 85bca31d-c3be-46b6-b330-b80648472378
md""" 
# System Hamiltonian of bipartite system
"""

# ╔═╡ ebd5efea-94b6-419e-8ff1-9dcf624ddaba
md"""
The system Hamiltonian for a bipartite system $\mathcal{H}_{\mathcal{A}} \otimes \mathcal{H}_{\mathcal{B}}$ is given by 

$H = \sum_{i = \{\mathcal{A}, \mathcal{B}\}} \varepsilon_i \hat{n}_i + \sum_{i=1}^3 \alpha_i (\mathbb{1}_{\mathcal{A}} \otimes {\sigma}_i) + \sum_{i=1}^3 \beta_i ({\sigma}_i \otimes \mathbb{1}_{\mathcal{B}}) + \sum_{ij=1}^3 \gamma_{ij} ({\sigma}_i \otimes {\sigma}_j)$

with Pauli matrices
$\sigma_1 = \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0  \end{pmatrix}$,
$\sigma_2 = \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0  \end{pmatrix}$,
$\sigma_3 = \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1\end{pmatrix}$



"""

# ╔═╡ c4406548-2119-480f-9438-22cd8c27e8f3
md"""
Coupling strength of interaction Hamiltonian ``H_I``
👉 $(@bind coupling Select(["normal" => "simple moderate coupling 🧾", "weak" => "weak coupling 🔗", "strong" => "strong coupling ⛓"]))
"""

# ╔═╡ 913d405a-e525-49d6-9021-227543b37953
md"""
# Initial state ``ρ(t_0)``
"""

# ╔═╡ a36e5515-e7f9-4c34-a48e-52a004d80f23
md"""
Switch from different initial states 👉 $(@bind initial_state Select(["thesis" => "Example in 📖", "product_state" => "Product state ♾", "choi_positive" => "Choi positive ➕", "back_in_time" => "Before product state 🕚", "s3b3" =>"3 x 3 🔱", "example" => "weak initial correlation 📍","slider"=> "Choose your own initial state 📐"]))
"""

# ╔═╡ 7479e663-e533-442c-8324-c60f4849c470
md""" 
# Dynamical map for different initial states
"""

# ╔═╡ 0872a46c-4f4b-49fa-8152-9193b71f332b
md""" 
## Dynamical map Λ
"""

# ╔═╡ c39065df-e7dc-4f09-b5c5-6066eec391ba
md"""
## Inverse of a dynamical map
"""

# ╔═╡ 88dcaaef-4782-4c64-acbe-91c9052cc955


# ╔═╡ b4e4baa5-338a-43b4-9035-d6be8e2d5e17
function pauli_coefficients(D)
	return [real(D[1,2]), -imag(D[1,2]), real(D[1,1]) - 0.5]
end

# ╔═╡ 19b181eb-8c95-41e2-9f4a-786cb5a25b8b
md"""
azimuthal angle 👉 $(@bind azim Slider(0:90, default = 30, show_value = true))

horizontal angle 👉 $(@bind hori Slider(0:90, default = 30, show_value = true))

"""

# ╔═╡ 38459aa7-557c-4252-a150-ba475a52a860
begin
n = 50
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);

x = cos.(u) * sin.(v)';
y = sin.(u) * sin.(v)';
z = ones(n) * cos.(v)';

# The rstride and cstride arguments default to 10
#plot(x[:],y[:],z[:])#, seriestype = :surface)
	
	#surface(x,y,z, fill_alpha = 0.5)
 surface(
  x,y,z,
  show_axis = false,
  color = fill(RGBA(1.,.5,1.,0.6), n, n),
 # limits = HyperRectangle(Vec3f0(-1.01), Vec3f0(1.01))
)
	
end

# ╔═╡ 455440b2-fc42-4db3-b9dd-eedd3406103a
ff(x,y) = (((x.^2 + y.^2) < 1) ? sqrt(1- x.^2 - y.^2) : NaN)

# ╔═╡ 67c8115a-eb59-4215-a23d-a00065baeaba
ff_(x,y) = (((x.^2 + y.^2) < 1) ? -sqrt(1- x.^2 - y.^2) : NaN)

# ╔═╡ 001db95a-4b60-483f-a863-7142269c9bbf
begin

surface(-1:0.1:1, -1:0.1:1, ff)
plot!(-1:0.1:1, -1:0.1:1,  ff_, seriestype = :wireframe)


end

# ╔═╡ 43b7be4e-9bae-4b1e-b536-4dd13aee7712
md"""
### Analysis of total time evolution superoperator
"""

# ╔═╡ 4754c50a-0d88-477d-a832-1c0eb8ccddac
md"""
Spectral analysis of total time evolution superoperator 

$\mathcal{U} = U^{\text{*}} \otimes U$

- ``\mathcal{U}`` is unitary (normal)
- ``\mathcal{U}`` is not Hermitian
- ``\mathfrak{C}_{\mathcal{U}} = \text{vec } U \cdot (\text{vec }U)^\dagger`` has rank 1, ``\lambda_1 = n``
"""

# ╔═╡ b7dfd6ac-6246-4655-a2dd-3d0e46e2b87e
md"""
### Time series of dynamical map
"""

# ╔═╡ 9a519b1a-6b0f-430c-869a-f8b902e862b6
md"""
### CP breaking of dynamical map with initial correlations in the beginning
"""

# ╔═╡ ea84b314-dfaf-4799-ac46-541b6a1318cd
# Examination why ``\Lambda(t, \varepsilon)`` is always not cp

# ╔═╡ d6619751-decf-40a4-874a-cbf0571de81c
@bind test_time Slider(0:0.00001:0.1, show_value = true)

# ╔═╡ be1e32d7-a48a-404c-b25c-e435a9b4c0ce
md"""
### Decomposition property
"""

# ╔═╡ 2cd1627f-6056-4c36-b089-c563097853f9


# ╔═╡ a8c70b91-6783-4507-9194-0a26f1e49860
md"""
Question: is the derivative always completely positive also for 

$\mathcal{K}(t,t_0) = \frac{\text{d}}{\text{d}\tau} \Lambda(\tau,t_0)\Big\rvert_{\tau = t}$
"""

# ╔═╡ 6d2b167f-ded7-4e6d-9d77-e2a555d1b46c
md"""
### Dynamical maps from generators
"""

# ╔═╡ c3696c66-4b84-40b6-af9f-d3c22ea84e35


# ╔═╡ 80f1a784-acf1-4df0-a36b-02665257143f
md"""
# Differential picture: Master equations and Lindblad forms
"""

# ╔═╡ 8f273160-c0ab-41e2-85ac-653db2ee7438
md"""
## Calculate Lindblad form (H, Γ, Aᵢ) from Kraus operators
This section illustrates how to derive from the **Kraus operators** ``\boldsymbol{K}_i`` of a CPT dynamical map the **Lindblad form**, consisting of the Lamb-shift Hamiltonian ``\boldsymbol{H}``, the positive decoherence matrix ``\boldsymbol{\Gamma}`` and the traceless orthonormal operator basis ``\boldsymbol{A}_i``

$\dot{\sigma}(t) = -\text{i} [\boldsymbol{H},\sigma(t)] + \sum_{ij} \Gamma_{ij}(t) \left(\boldsymbol{A}_i \sigma(t) \boldsymbol{A}_j^\dagger - \frac{1}{2} \{\boldsymbol{A}_j^\dagger \boldsymbol{A}_i, \sigma(t)\} \right)$
"""

# ╔═╡ b9bc1429-bd9a-452b-9ddf-34b91217de96
md"
$\text{Tr} \boldsymbol{A}_i = 0, \quad \text{for } i\geq 2$

$\langle \boldsymbol{A}_i, \boldsymbol{A}_j \rangle_{\text{HS}} = \delta_{ij}$
"

# ╔═╡ 06883a55-b99c-4edd-ae8b-e9cc9abe2b99
md"""
Check if derivative of dynamical map ``\Lambda(t,t_0)`` at time ``t_1 > t_0``  is obtained by the derivative of ``\Lambda(t,t_1)`` at time ``t_1``


"""

# ╔═╡ 3553ea01-ba81-48cd-be9f-886508a70a2d
	#A = [zeros(ComplexF64,4,4) for _=1:16]
	#A[1] = I(4)/sqrt(4)
	#A[2:16] = [unity(i,4) for i = 2:16]
	#L = [real(tr(A[i])) .> 0 for i = 1:16]
	#for i = [6,11,16]
	#	A[i] = (A[i] - I(4)/4)*sqrt(2/3)
	#end

# ╔═╡ eaafb3a3-a12b-4c5a-947a-2bcd376612c0


# ╔═╡ 022cc2fa-f6d4-4fec-b7ac-645890984d46
md"""
### Direct derivative of dynamcial map
Deal with 

$\frac{d}{dt} \mathfrak{C}\Big(\exp(-i (H_S \otimes \mathbb{1} + \mathbb{1} \otimes H_B + H_I)  t)\Big)$

"""

# ╔═╡ 558e2f3a-2942-4062-982e-f5c94c0f9588
begin
# choi(A ⊗ᶜ B) = A[:] ⋅ B[:]ᵗ
# exp(A ⊗ I) = exp(A) ⊗ I


md"""
Note that

$\exp(A \otimes \mathbb{1}) = \exp(A) \otimes \mathbb{1}$
and that

$\mathfrak{C}(A \overset{c}{\otimes} B) = \text{vec} A \cdot (\text{vec} B)^T$

Decompose equation above to

$\frac{d}{dt} \mathfrak{C}\Big(\exp(-i (H_S \overset{c}{\otimes} \mathbb{1} )  t)\Big) = \text{vec} \Big[-iH_S \cdot \exp(-iH_S t)\Big] \cdot (\text{vec } \mathbb{1})^T$
"""
end

# ╔═╡ 878a884d-196c-4403-bc86-9e642956441c
md"""
Next step: decompose exponential, especially think of how to decompose $H_I$
"""

# ╔═╡ 5e1377b7-c784-4e42-8190-ffbf48f0ecc1
md""" 
##  Initial classical-quantum correlated state
"""

# ╔═╡ 822ea5ab-4a3a-45ee-903c-ff9bad9d5f16
md""" 
## Example of non--positive dynamical map
"""

# ╔═╡ 4cdab1a4-1784-4430-ae4f-50a4459d83a4
md"""
# Example of a non-accessible map
The transposition superoperator

"""

# ╔═╡ a558b847-3f09-473c-9ec5-70badcda7589
md"""
# Appendix: Functions and Initializations
"""

# ╔═╡ 8407fc8c-95b4-4536-bbda-b3f4945518b0
TableOfContents()

# ╔═╡ 7f43573f-010f-455a-82c7-f2f509cce3d3
begin
	time_slider = @bind time Slider(0:0.001:20, default = 1, show_value = true)
	inter_time_slider = @bind inter_time Slider(0:0.0001:1, default = 0, show_value = true)
	md""" ## Interactive widgets
	Definition of Sliders"""
end

# ╔═╡ f850340c-7a18-4693-8822-7ef29e3d2675
begin
 md"""
 Intermediate time to examine new dynamical map 👉 $(inter_time_slider)
 """


end

# ╔═╡ 95c1685e-039b-492b-8a8c-d2914036addb
time_slider

# ╔═╡ c2af9765-647f-4f25-adc9-139f7333413c
inter_time_slider

# ╔═╡ e8c989d6-9ba6-40d1-ad36-d768172d5f5e
time_slider

# ╔═╡ 66b53661-7f06-4290-94a1-28240e29f067
time_slider

# ╔═╡ 9e17a64b-4b49-44fb-8b5d-b409d6897de1
inter_time_slider

# ╔═╡ 754b5aaf-841c-445d-ad8f-06a8e7ed391d
inter_time

# ╔═╡ 002504f4-e74a-4e8c-80fa-f4c5f2e90df0
time_slider

# ╔═╡ d28b8274-7435-484a-bde7-61443d7e4ae2
begin
	scrub_1 = @bind s1 Scrubbable(-0.5:0.01:0.5, format=".03")
	scrub_2 = @bind s2 Scrubbable(-0.5:0.01:0.5, format=".03")
	scrub_3 = @bind s3 Scrubbable(-0.5:0.01:0.5, format=".03")
	scrub_4 = @bind s4 Scrubbable(-0.5:0.01:0.5, format=".03")
	scrub_5 = @bind s5 Scrubbable(-0.5:0.01:0.5, format=".03")
	scrub_6 = @bind s6 Scrubbable(-0.5:0.01:0.5, format=".03")
	scrub_7 = @bind s7 Scrubbable(-0.5:0.01:0.5, format=".03", default = 0.25)
	scrub_8 = @bind s8 Scrubbable(-0.5:0.01:0.5, format=".03")
	scrub_9 = @bind s9 Scrubbable(-0.50:0.01:0.50, format=".03")
	
	scrub_a1 = @bind a1 Scrubbable(-0.50:0.01:0.50, format=".03", default = -0.5)
	scrub_a2 = @bind a2 Scrubbable(-0.50:0.01:0.50, format=".03")
	scrub_a3 = @bind a3 Scrubbable(-0.50:0.01:0.50, format=".03")
	
	scrub_b1 = @bind b1 Scrubbable(-0.50:0.01:0.50, format=".03", default = -0.5)
	scrub_b2 = @bind b2 Scrubbable(-0.50:0.01:0.50, format=".03")
	scrub_b3 = @bind b3 Scrubbable(-0.50:0.01:0.50, format=".03")
	
	scrub_i1 = @bind i1 Scrubbable(-0.50:0.01:0.50, format=".03", default = -0.5)
	scrub_i2 = @bind i2 Scrubbable(-0.50:0.01:0.50, format=".03")
	scrub_i3 = @bind i3 Scrubbable(-0.50:0.01:0.50, format=".03")
	
	md" Definition of Scrubbables"
end

# ╔═╡ fc12024b-2785-4f89-a457-95a2deb5045f
begin
	if initial_state == "slider"
		md"""
		You can choose the parameters of the initial state:
		
		$\rho(t_0) =\frac{1}{4}\mathbb{1}_{\mathcal{A}} \otimes \mathbb{1}_{\mathcal{B}}+ \sum_{i=1}^3 \alpha_i (\mathbb{1}_{\mathcal{A}} \otimes {\sigma}_i) + \sum_{i=1}^3 \beta_i ({\sigma}_i \otimes \mathbb{1}_{\mathcal{B}}) + \sum_{ij=1}^3 \gamma_{ij} ({\sigma}_i \otimes {\sigma}_j)$
		
		``\alpha_1`` 👉 $(scrub_a1),  ``\alpha_2`` 👉 $(scrub_a2), ``\alpha_3`` 👉 $(scrub_a3)
		
		``\beta_1`` 👉 $(scrub_b1),  ``\beta_2`` 👉 $(scrub_b2), ``\beta_3`` 👉 $(scrub_b3)
		
		``\gamma_{11}`` 👉 $(scrub_1),  ``\gamma_{12}`` 👉 $(scrub_2), ``\gamma_{13}`` 👉 $(scrub_3)
		
		``\gamma_{21}`` 👉 $(scrub_4),  ``\gamma_{22}`` 👉 $(scrub_5), ``\gamma_{23}`` 👉 $(scrub_6)
		
		``\gamma_{31}`` 👉 $(scrub_7),  ``\gamma_{32}`` 👉 $(scrub_8), ``\gamma_{33}`` 👉 $(scrub_9)
		
		"""
		
		
		
	end
end

# ╔═╡ 1e4adc4d-52cf-41df-ad62-cab016f55397
[scrub_i1, scrub_i2, scrub_i3]

# ╔═╡ 0c09a3b6-920d-4732-af67-86b36bd88292
scrub_a1, scrub_a2, scrub_a3

# ╔═╡ dc95c17d-de0f-4b72-b57e-0dab473a69cd
[scrub_1,scrub_2,scrub_3,scrub_4, scrub_5, scrub_6, scrub_7, scrub_8, scrub_9]

# ╔═╡ 5c9ac0b7-11e7-4f52-90ff-04ce131c61d8
begin
σ = [[0 1; 1 0],  [0 -im; im 0],  [1 0; 0 -1]]
	
σ_dim = [σ, σ, [[0 1 0; 1 0 0; 0 0 0], [0 -im 0; im 0 0; 0 0 0], [1 0 0; 0 -1 0; 0 0 0], 
	 [0 0 1; 0 0 0; 1 0 0], [0 0 -im; 0 0 0; im 0 0], [0 0 0; 0 0 1; 0 1 0], 
	 [0 0 0; 0 0 -im; 0 im 0], 1/sqrt(3) * [1 0 0; 0 1 0; 0 0 -2]]
]
end

# ╔═╡ 5663ed51-a921-4a8b-b351-86aca24420b6
function  qu_bit_system(epsilon  = 1, 
			alpha = [0,0,0], 
			beta= [0,0,0], 
			gamma = zeros(3,3), dim_1 = 2, dim_2 = 2)
			
	return H = epsilon * I(dim_1) ⊗ I(dim_2) +
		sum([(I(dim_1) ⊗ σ_dim[dim_2][i]) * alpha[i] for i = 1:length(σ_dim[dim_2])]) + 
		sum([(σ_dim[dim_1][i] ⊗ I(dim_2)) * beta[i] for i=1:length(σ_dim[dim_1])]) + 
		sum([gamma[i,j] * (σ_dim[dim_1][i] ⊗ σ_dim[dim_2][j]) for i=1:length(σ_dim[dim_1]), j=1:length(σ_dim[dim_2])])
	end

# ╔═╡ 3140729e-d516-434d-ba32-3ff9705ee83b
begin
keep_working(text=md"The answer is not quite right.", title="Keep working on it!") = Markdown.MD(Markdown.Admonition("danger", title, [text]));

almost(text, title="Almost there!") = Markdown.MD(Markdown.Admonition("warning", title, [text]));

hint(text, title ="Hint") = Markdown.MD(Markdown.Admonition("hint", title, [text]));
	
correct(text=md"Great! You got the right answer! Let's move on to the next section.", title="Got it!") = Markdown.MD(Markdown.Admonition("correct", title, [text]));
md" Definition of Boxes"
end

# ╔═╡ 75ebae9f-1c69-4d92-9cdf-a0f4b0971576
begin



	
	
	function density_pauli(α)
		return α[1]/2 * I(2) + sum([α[i+1] * σ[i] for i = 1:3])
	end
	
	function density_matrix(n, neg = 0, real_valued = false)
		#trace 1 remains preserved!
		dia = rand(n)
		if neg > 0
			dia[rand(1:n, neg)] .*= -1
		end
		dia = dia/sum(dia)
		if real_valued
			temp = rand(n,n)
		else
			temp = rand(n,n) + rand(n,n) * im
		end
		ew, ev = eigen(Hermitian(temp'*temp))
		d_mat = Hermitian(ev* Diagonal(dia) * ev')
		return d_mat
	end
	
	function unitary(n)
		A = (rand(n,n) .-0.5) + im*(rand(n,n) .-0.5)
		return eigvecs(A'+A)
	end
	
	
	function unity(i,n)
		E = spzeros(Bool,n,n)
		E[i] = true
    	return E
	end
	function partial_trace_matrix(dim_1, dim_2, traced_system, order="lex")
# traces out the system indicated by traced_system
# for C = kron(A,B), tracing over the first system (traced_system = 1); 
# yields: reshape(p1*C[:], 4,4)/tr(A)  == B
# 
# when colex ordered:
# flip traced_system
		
		if order == "colex"
			traced_system = 3 - traced_system 
		end
		
		if traced_system == 1
			T = [kron(sparse(I, dim_1,dim_1), unity(i,dim_2))[:] for i = 1:dim_2^2]
			col = reduce(vcat,[findnz(T[i])[1] for i = 1:dim_2^2])
			row = kron(1:dim_2^2, ones(Int64,dim_1))
			return sparse(row, col, trues(dim_2^2*dim_1))
		else
			T = [kron(unity(i, dim_1), sparse(I, dim_2, dim_2))[:] for i = 1:dim_1^2]
			col = reduce(vcat, [findnz(T[i])[1] for i = 1:dim_1^2])
			row = kron(1:dim_1^2, ones(Int64,dim_2))
			return sparse(row, col, trues(dim_1^2*dim_2))
			#return    Tr_B = [reshape(kron(reshape(unity(i,dim_1), dim_1, dim_1), I(dim_2)), 1, dim_2^2 * dim_1^2)[1,j] for i = 1:dim_1^2, j =1:dim_1^2*dim_2^2]
		end
	end


	function choi(A, dim_1)
		mm, nn = size(A)
		n = mm ÷ dim_1
		return reshape(permutedims(reshape(A,n,dim_1,n,dim_1),(1,3,2,4)), mm,mm)
	end

	function choi_rect(A,dim_1, dim_2, dim_3, dim_4)
		return reshape(permutedims(reshape(A, dim_1, dim_2, dim_3, dim_4), (1,3,2,4)), dim_1*dim_3, dim_2*dim_4)
	end
	
	
	function choi_index(i, n)
		return  (i.-1) .% n[1] .+ 1 .+
		n[1] .* ((i.-1) .% prod(n[1:3]) .÷ prod(n[1:2]) .+ 
		n[3] .* ((i.-1) .% prod(n[1:2]) .÷ n[1] .+
		n[2] .* ((i.-1) .÷ prod(n[1:3]))))
	end

	function choi2kraus(C,difference_map = false)
		if norm(C - C') > 10^-10
			#error("Choi matrix is not Hermitian!")
			return almost(md"Choi Matrix is not Hermitian!", "Warning")
		end
		 eww,evv = eigen(Hermitian(C))

		if (difference_map == false) & (any(eww .< 0))
			return keep_working(md"Eigenvalues of Choi matrix are not positive!", "Error")
			#error("Eigenvalues of Choi matrix are not positive!")
		end
		n = Int(sqrt(size(C,1)))
		K = 2
		K = [reshape(evv[:,k], n,n) * sqrt(abs(eww[k])) for k = 1:n^2]

		if difference_map == true
			ep = sign.(eww)
			return K, ep
		else 
			return K
		end
	

	end
		

	function ptr(ρ, dim_1, traced_system = 1, order="lex")
		# get partial trace superoperator
		N,M = size(ρ)
		dim_2 = N÷dim_1
		PT = partial_trace_matrix(dim_1, dim_2, traced_system, order) 
		new_dim = traced_system == 1 ? dim_2 : dim_1
		# apply superoperator to vectorized density matrix and reshape to new dimensions
		σ = reshape(PT * ρ[:], new_dim, new_dim)
		return σ
	end

	function factorize(A, dim_1, dim_2, order="lex")
		σ = ptr(A, dim_1, 2, order)
		ρ = ptr(A, dim_1, 1, order)
		A_tensor_prod = (σ ⊗ ρ)
		rho_c = A - A_tensor_prod
		return A_tensor_prod, rho_c
	end

	function normed_vector(k)
		v = (rand(k) .- 0.5) + im*(rand(k) .- 0.5)
		return v/norm(v)
	end

	function prob_dist(k)
		p = rand(k)
		return p/sum(p)
	end
	
	function PPT(rho,dim_1)
		dim_2 = size(rho,1)÷dim_1
		if dim_1 *dim_2 > 6
			error("positive partial trace does not work for too large dimensions")
		end
		return eigvals(reshape(permutedims(reshape(rho,dim_1,dim_2,dim_1,dim_2), [1,4,3,2]),dim_1*dim_2, dim_1*dim_2))

	end
	
	function sep_state(k=1,n=2)
		VVV = sum([(normed_vector(n) ⊗ normed_vector(n)) for _=1:k] .* (prob_dist(k)))
		return VVV / norm(VVV)
	end
	
	
	#test for k-block positivity
	# strategy: generate random matrices in 2x2 and test whether Choi matrix maps to positive operators, if true for all, then positive.
	# alternatively: check k-positivity: k as optional parameter. Create states with Schmidt rank k
	#Schmidt rank: 
	function pos_test(choi, k = 1, n = 2)
	
		runs = 10000
		test = zeros(runs)
		vector = zeros(ComplexF64, n^2)
		for i = 1:runs
			
			#sum([density_matrix(2) for _=1:k] * p_rand/sum(p_rand))

			VVN = sep_state(k,n)

			test[i] = real(VVN'* choi *VVN)

			if real(test[i]) < 0
				vector = VVN
			end
		end

		return sum(test .> 0) / runs
	
	end
	
md"""
# Functions
"""
end

# ╔═╡ c8e02ee8-2c21-4a37-9875-c4a463e0b462
begin
	ϵ = [1,2]
	α = [0,0,0]
	β = [0,0,0]
	if coupling == "normal"
		γ = [1 0 0; 0 0 0; 0 0 0]
	elseif coupling == "weak"
		γ = 0.1*[1 1 1; 1 1 1; 1 1 1]
	elseif coupling == "strong"
		γ = [10 20 10; -15 10 40; 23 12 30]
	end
	
	if initial_state == "s3b3"
		ϵ = [1,2,4]
		α = [0,0,0,0,0,0,0,0,0]
		β = [0,0,0,0,0,0,0,0,0]
		γ = unity(1,9)
		H = I(3) ⊗ Diagonal([0,ϵ[2], ϵ[3] + ϵ[2]]) + 
		Diagonal([0,ϵ[1], ϵ[3] + ϵ[1]])⊗ I(3) +	
		5*qu_bit_system(0,α, β, γ,3,3)   
	else
		H = qu_bit_system(0, α, β, γ) + I(2) ⊗ Diagonal([0,ϵ[2]]) + 
		Diagonal([0,ϵ[1]]) ⊗ I(2)
	
		
	end
end

# ╔═╡ 872b503b-4c40-4944-aefb-e52c559c72d5
eigvals(exp(-im*H*time))

# ╔═╡ 98c23700-c267-4c40-9a45-f5bb382c8429
eigvals(conj.(exp(-im*H*time)))

# ╔═╡ a5e38ca3-8184-4160-a6b1-4c1783c6e185
begin
	dim_1 = 2
	dim_2 = 2
	if initial_state == "thesis"
		rho_0 = qu_bit_system(1/4, [0.0, 0.00, +0.0], 
							  [0.00, -0.0, -0.0], [0 0 0; 0 0 0; 1/4 0 0]) 
	elseif initial_state == "back_in_time"
		eps_time = -0.2
		rho_0 = Hermitian(exp(-im*H*eps_time) * kron(I(2)/2 + σ[2]/3, I(2)/2 + σ[3]/4) * exp(im * H *eps_time))
	elseif initial_state == "example"
		rho_0 = qu_bit_system(1/4, [0.00001, .010, +0.0], 
							  [0.001, 0.00001, -0.0], [0 0.0001 0; 0 0 0.0001; 0.001 0 0]) 
	elseif initial_state == "product_state"
		rho_0 = kron(I(2)/2 + σ[1]/3, I(2)/2 + σ[1]/4)
	elseif initial_state == "s3b3"
		rho_0 = 1 * kron(I(3)/3 + [0 im 1; -im 0 0; 1 0 0]/5, I(3)/3+ 1/4 * [0 0 1; 0 0 0; 1 0 0])# + 
				#2/3 * kron(I(3)/3+  [0 im 1; -im 0 0; 1 0 0]/5, I(3)/3) 
		dim_1 = 3
		dim_2 = 3
		
	elseif initial_state == "choi_positive"
		rho_0 = [0.35304-0.0im          0.0+0.0im          0.0+0.0im      -0.03293-0.06263im;
      0.0+0.0im      0.32484-0.0im      0.09992+0.00626im       0.0+0.0im;
      0.0+0.0im      0.09992-0.00626im  0.17516-0.0im           0.0+0.0im;
 -0.03293+0.06263im      0.0+0.0im          0.0+0.0im       0.14696+0.0im]

	else
		rho_0 = qu_bit_system(1/4, [a1, a2, a3], 
								[b1, b2, b3], [s1 s2 s3; s4 s5 s6;s7 s8 s9])
	end
	
	
	# alternative
	Dim = 2
	RHO_b = [density_matrix(Dim) for _=1:Dim]
	RHO_s = density_matrix(Dim)
	evals, Pro = eigen(RHO_s)
	
	prob = prob_dist(Dim)
	
	#rho_0 = sum([kron(Pro[:,i] * Pro[:,i]', RHO_b[i]) for i = 1:Dim] .*prob)
	
	rho_t_0, rho_c = factorize(rho_0,dim_1,dim_2, order)
	md""" 
	Definition of $\rho_0$
	"""
end

# ╔═╡ aedb8227-6f24-43ae-9634-32cf365ccdcc
norm(γ) /norm(rho_c)

# ╔═╡ 4f87569b-777e-4e97-ac80-8b7b76115d2a
rho_0

# ╔═╡ 5afc42f7-3253-438d-96a8-d665a1b9a1cd
if any(eigvals(rho_0) .< 0)
	keep_working(md"Tune the parameters such, that the initial state becomes positive: ``\rho(t_0)> 0``.
		
The current eigenvalues are: 
		$(string(round.(eigvals(rho_0), digits = 2)))" ,"Initial state  not valid!")
end

# ╔═╡ b595349b-b572-4cc8-92b1-07631cc8fad7
eigvals(rho_0) # eigenvalues of the initial state

# ╔═╡ 49b3aeaf-ff2b-4805-bf4b-58aed5c49607
rho_0

# ╔═╡ 8cd0438c-c480-4b73-93af-79e3f3a5dfa9
eigvals(rho_0)

# ╔═╡ 596a9128-2fff-47e8-8b6f-e9fd8686a279
[ptr(rho_0,dim_1,2,order), ptr(rho_0,dim_1,1,order)] #σ(t_0) and ρ_B(t_0)

# ╔═╡ c578581e-467e-482c-9cce-2ad55d32267f
PPT(rho_0,dim_1) #initial state is separable, but is it classical quantum correlated?

# ╔═╡ 8c779399-7db0-421e-b1b2-659d28fc8c41
begin 
	#using the product state rho_t_0
	rho_B = ptr(rho_t_0,dim_1,1, order) # the bath denstiy matrix
	sigma_0 = ptr(rho_t_0,dim_1,2, order) # the system density matrix
	compare = sigma_0 ⊗ rho_B - rho_t_0 #check if this is really a product state

	U = exp(-im*H*time)
	#NOTE: the choi matrix notation needs a transpose the match the lexicographical order
	if order == "colex"
		Λ = choi(choi(U,dim_1) * kron(rho_B, I(dim_2)) * choi(U,dim_1)',dim_1)#
	else
		Λ = transpose(choi(transpose(choi(U,dim_1)) * kron(I(dim_1), rho_B) * transpose(choi(U,dim_1))',dim_1)) # this is a real Kronecker product
	end

	Λ_c = ptr(U * rho_c * U', dim_1, 2, order)[:] * collect(I(dim_1)[:])'

	test = density_matrix(dim_1)
	dyn_1 = reshape(Λ * test[:], dim_1,dim_1)



	dyn_2 = ptr(U * (test ⊗ rho_B) * U',dim_1,2, order)
	dyn_1 - dyn_2
end

# ╔═╡ 9a7de167-8474-4a22-99bf-66af8bffa736
abs.(eigvals(Λ + Λ_c))

# ╔═╡ 9330d1cf-e0e1-41c1-a876-5fae11633ce8
Λ_c + Λ

# ╔═╡ 4f8ada04-10a7-4783-8709-db55428824b3
sum(eigvals(Λ + Λ_c))

# ╔═╡ 15cb7806-8b8a-4920-9197-18dd80a67926
sigma_0

# ╔═╡ 5d00c2ec-77c9-49a6-8020-a1919ce99bbb
sigma_0

# ╔═╡ 2cab05f7-bc59-4d26-a535-495b5ab104c0
begin
	sig_n = 1/2 * I(2) + i1 * σ[1] + i2 * σ[2] + i3*σ[3]
	#pure state: a1 = -0.5
	eigvals(sig_n)
	#sig_n is not compatible:
	eigvals((sig_n ⊗ rho_B) + rho_c)
end

# ╔═╡ faf456bc-9145-4d87-ad4d-6ecc8c78bd41
eigvals(reshape((Λ + Λ_c)* sig_n[:],2,2))
# yields negative result

# ╔═╡ e63d17ba-767c-4136-8dc3-3da609f8dda5
reshape((Λ + Λ_c)* sig_n[:],2,2)

# ╔═╡ 1d31452b-5ccf-49ba-ac94-d5c28febe295
sum(eigvals(Λ + Λ_c))

# ╔═╡ a3ace2bc-5263-4868-a999-098f70e19483
begin
Λ₋ = (Λ + Λ_c)^-1
# hermtiticity preserved, not complete positive!, 
	eigvals(choi(Λ₋,dim_1))

pos_test(choi(Λ₋,dim_1))
	# not positive
end

# ╔═╡ 3006719e-ed10-4186-ade1-7c474af59883
eigvals(reshape(Λ₋ * density_matrix(2)[:], dim_1, dim_1))

# ╔═╡ a3c9bf88-4404-4a6c-9c18-f2e58c325729
begin
#map randomly created density matrices: find Pauli representations
R = [density_matrix(2) for _=1:1000]
	#RR = reduce(hcat, pauli_coefficients.(R))
delta = 0.05

R = 	[I(2)/2 + σ[1] * i + σ[2] * j + σ[3] * k for i= -0.5:delta:0.5, j= -0.5:delta:0.5, 
k = -0.5:delta:0.5]
sig_inv = pauli_coefficients.([reshape(Λ₋ * R[i][:],2,2) for i = 1:length(R)])
RR = reduce(hcat, sig_inv)
rad = sqrt.(sum(reduce(hcat,pauli_coefficients.(R)).^2,dims = 1)[:])
RR = RR[:,rad .< 0.5]
rad = rad[rad .< 0.5]
end

# ╔═╡ 10a8a0e7-a7e0-48ba-a177-6df6cbeb5f78
size(RR)

# ╔═╡ 9863549d-e86e-4b0c-9c18-804b233cea89
reduce(hcat, pauli_coefficients.(R))

# ╔═╡ 7835a031-cffe-4d47-a47c-d8631f98ef2a
[reshape(Λ₋ * R[i][:],2,2) for i = 1:length(R)]

# ╔═╡ 254be7e0-d60a-45dc-8113-46fffe2cbd88
 rad .< 0.5

# ╔═╡ 4dae271b-27b9-460e-988c-d733d8d79e87
begin
plt2 = plot3d(
    1,
    xlim = (-0.6, 0.6),
    ylim = (-0.6, 0.6),
    zlim = (-0.6, 0.6),
    title = "Lorenz Attractor",
    marker = 2,
)

#Blochsphere
upper(x,y) = 0.5 > x.^2 + y.^2 ? sqrt(0.5 - x.^2 - y.^2) : NaN

scatter3d(RR[1,:], RR[2,:], RR[3,:], markerz = (rad .-minimum(rad)), m = (3, .3, :bluesred, Plots.stroke(0)), leg = false, cbar = true, w = 0, camera = [azim,hori], axis = :equal, xlabel = "σ₁", ylabel = "σ₂", zlabel = "σ₃")

end

# ╔═╡ 2921da97-0099-4662-9d7e-4a8696d85de5
upper(.1,0.2)

# ╔═╡ db14e9c5-2b72-4f8d-b1bc-c2f9d11b67a1
function dynamical_map(t, rho_0 = rho_0, Ham = H)

	rho_B = ptr(rho_0,dim_1,1,order)
	sigma_0 = ptr(rho_0, dim_1,2,order)

	rho_c = rho_0 - sigma_0 ⊗ rho_B

	U = exp(-im * Ham * t)
	if order == "colex"
		Λ_0 = choi(choi(U,dim_1) * kron(rho_B, I(dim_2)) * choi(U,dim_1)',dim_1)#
	else
		Λ_0 = transpose(choi(transpose(choi(U,dim_1)) * kron(I(dim_2), rho_B) * transpose(choi(U,dim_1))',dim_1)) # this is a real Kronecker product
	end
	Λ_c = ptr(U * rho_c * U', dim_1, 2, order)[:] * collect(I(dim_1)[:])'
	Λ = Λ_0 + Λ_c
	return Λ

end

# ╔═╡ e20411a0-c43f-44c5-8a60-252348ef9501
begin
	#run time evolution for different times
	t_time = 0:0.001:1
	D =  dynamical_map.(t_time)
	ews = real.(eigvals.(choi.(D,dim_1)))
	eig_vector = reduce(hcat,ews)

	sigma_time = [reshape(D[i] * sigma_0[:],dim_1,dim_1) for i = 1:length(t_time)]
	sigma_vec = [real.(reduce(hcat,diag.(sigma_time))) ; real(reduce(hcat,diag.(sigma_time,1))); 	imag(reduce(hcat,diag.(sigma_time,1)))]
	sigma_eig = real.(reduce(hcat,eigvals.(sigma_time)))
end

# ╔═╡ 3b7400dd-1c04-47f6-9f65-52bdedb63002
round.(choi(D[2],dim_1),digits=7)

# ╔═╡ 580ff266-927a-447a-9a48-9e64bf07b806
real.(tr.(sqrt.(D.*transpose.(conj.(D)))))

# ╔═╡ 00ed1878-0fc5-464e-8f9c-66484f4f369e
# trace norm
[sort(sqrt.(eigvals(D[40]*D[40]'))) sort(svdvals(D[40]))]
# compare with singular values

# ╔═╡ 40b50286-d099-41a1-b73c-167625b0221c
begin
plot(t_time, sigma_vec', label=["σ₁₁" "σ₂₂" "Re(σ₁₂)" "Im(σ₁₂)"], title="Reduced time evolution, analysis of σ")
plot!(t_time, sigma_eig',linestyle=:dash, label=["λ₁" "λ₂"])
end

# ╔═╡ 994c73ac-d001-4930-b28a-705ea98cada7
eigvals(choi(D[1],dim_1))

# ╔═╡ 908235d9-f919-4d89-a9d8-d172f0bce524
begin
	H_test = rand(dim_1 * dim_2,dim_1 * dim_2) + rand(dim_1 * dim_2,dim_1 * dim_2) * im

	H_test = H_test + H_test'
eigvals(H_test)
	U_test = [exp(-im * H_test * i) for i = t_time]
# U_test () U_test'

	U_cal = [kron(conj.(U_test[i]), U_test[i]) for i = 1:length(U_test)]
	# U_cal is unitary!
	# U_cal is normal but not Hermitian
	

ew_test = sum(reduce(hcat, [eigvals( U_cal[i] ) for i = 1:length(U_test)]), dims = 1)

svd_test = sum(reduce(hcat, [svdvals( U_cal[i] ) for i = 1:length(U_test)]), dims = 1)

	C_Ucal = [choi(U_cal[i], dim_1*dim_2) for i = 1:length(U_test)]
	ew_choi_test = maximum(reduce(hcat,eigvals.(Hermitian.(C_Ucal))), dims = 1)

	
	
end

# ╔═╡ 13d983b3-d9af-4360-81c9-2fbc91907a2a
begin
plot(t_time, real.(ew_test)')
plot!(t_time, abs.(svd_test)')
plot!(t_time, ew_choi_test')

end

# ╔═╡ 4c85b70f-7d65-449e-9c17-9db3414e918a
begin
	# intermediate time evolution'
inter_time
t_inter_time = inter_time:0.001:maximum(t_time)
U_inter = exp(-im * H * inter_time)
rho_t_inter = U_inter*rho_0*U_inter'

D_inter = map(t -> dynamical_map(t, rho_t_inter), t_inter_time .- inter_time)
ews_inter = real.(eigvals.(choi.(D_inter,dim_1)))
eig_vector_inter = reduce(hcat,ews_inter)
end

# ╔═╡ 48677999-941a-437c-90c2-e35ef44ee5f0
(round.(rho_t_inter,digits=5))

# ╔═╡ 50eca964-09a9-48ed-9b75-d3ac83e8b1d7
f = t -> dynamical_map(t, rho_t_inter)

# ╔═╡ 17264c91-b88a-4e92-a336-3695e4263037
f(3)

# ╔═╡ f80055ca-bea7-4ef3-80c2-a8892b374df3
dynamical_map(0.1)

# ╔═╡ 8393a842-397c-4600-bbca-2345ea06722e
begin
# get master equation operator

	method = FiniteDifferenceMethod(1:5, 1)
#dynamical_map
round.(central_fdm(5,1)(dynamical_map,0), digits=10)
	
end

# ╔═╡ f29ab0fa-72f6-49a4-abc2-492a32a88374
begin

	D_01 = dynamical_map(inter_time)
	D_02 = dynamical_map(inter_time + 1)
	D_12 = dynamical_map(1,rho_t_inter)
	central_fdm(5,1)(t -> dynamical_map(t, rho_t_inter), 0)

	D_10 = dynamical_map(-inter_time,rho_t_inter)

end

# ╔═╡ d170f96b-f329-482e-be03-0cbfc75c41dc
(D_12 * D_01   - D_02 )#* sigma_0[:]

# ╔═╡ 884c70c1-defe-4680-a575-6915a2917abc
dynamical_map(-1, rho_t_inter)

# ╔═╡ 6a2eecf5-998f-44b2-8576-db0092e706d9
central_fdm(5,1)(t -> dynamical_map(t, rho_0), inter_time)

# ╔═╡ 5d450f8d-c8c4-4a58-b2a6-97bfe109f91f
begin
	#check
	
	norm(ptr(U*rho_0*U',dim_1,2, order)- reshape((Λ+Λ_c) * ptr(rho_0,dim_1,2, order)[:],dim_1,dim_1))
	
end

# ╔═╡ f617e5dc-f2f7-4d79-8258-b3c95de86653
C_0 = choi(exp(-im*H*test_time),dim_1) * kron(rho_B, I(dim_2)) * choi(exp(-im*H*test_time),dim_1)'

# ╔═╡ 637a2af1-d61a-4dff-aa30-f45c4ac95f6a
C_0

# ╔═╡ a2747d76-9f86-4bad-87a7-e71b9a24cbe8
U_t = choi(exp(-im*H*test_time),dim_1)

# ╔═╡ 3239e943-5328-41fd-9435-798504eb34a8
real.(round.(U_t, digits = 1))

# ╔═╡ 7179873b-53d2-44d4-a618-63ca58293b8d
real.(round.(U_t * kron(density_matrix(dim_1), I(dim_2))*U_t',digits = 2))

# ╔═╡ 59ee6550-c3ec-4874-8afe-e852a25d15b6
C_c = 0.0 * round.(kron(I(dim_1), ptr(exp(-im*H_test*test_time) * rho_c * exp(im*H_test*test_time),dim_1,2,order)), digits = 6)

# ╔═╡ 79a90d41-8846-49ac-8dc5-e615a9c26e38
real.(eigvals(C_0 + 0* C_c))


# ╔═╡ f7915998-5d2b-4fcb-bccf-c2163d9dc71b
CU = (choi(exp(-im*H_test * test_time),dim_1))

# ╔═╡ 1044ff8d-3778-4f69-93ec-fe4335cba189
ptr(C_0,dim_1,1,order)

# ╔═╡ eae36bd4-9733-43b1-8375-f1b3eab87aa0
eigvals(ptr(exp(-im*H_test*test_time) * rho_c * exp(im*H_test*test_time),dim_1,2,order))

# ╔═╡ b19b685a-edb2-4913-b21f-68015b961911
begin
t_21 = time - inter_time
DD_21 = map(t -> dynamical_map(t, rho_t_inter), t_21)
DD_10 = dynamical_map(inter_time, rho_0)
DD_20 = dynamical_map(time, rho_0)
test_sig = sigma_0 * 0.999 + 0.001 * density_matrix(2)
end

# ╔═╡ 62b3f7c0-5b11-4984-b433-55b6bd40b8a7
DD_21 * DD_10 - DD_20

# ╔═╡ 7e319790-26ac-4409-8d5d-ebc2962a87da
abs.(eigvals(DD_21 * DD_10 - DD_20))
# the dimension of the kernel of DD_21 * DD_10 - DD_20 is one corresponding to the initial value sigma_0 that set up DD_21
# so the decomposition property holds only for the initial value sigma_0

# Question: can one construct a composing family of dynamical maps just from the generators? -> turn generators into dynamical maps!

# ╔═╡ 7a6ea79a-8c86-4bd7-ab36-92b9ccbe2289
begin
# basis for traceless operators
	A = [zeros(ComplexF64,2,2) for _=1:4]
	A[1] = I(2)/sqrt(2)
	A[2:4] = [unity(i,2) for i = 2:4]
	
	for i = [4]
		A[i] = (A[i] - I(2)/2).*sqrt(2)
	end
end

# ╔═╡ 08606417-ef27-4944-afe1-42cdba38e2e4
tr.(A)

# ╔═╡ cc05e609-805a-47de-8824-dc446d2676e2
[tr(A[i]*A[i]') for i = 1:4]

# ╔═╡ 038aaf32-fccc-45b3-a586-248ab7b7e45f
begin 
	D_inv = D_01^-1
	C_inv = choi(D_inv,dim_1)
	eigvals(C_inv)

end

# ╔═╡ ffa20581-01a4-44c4-b89e-a4732243522c
D_inv

# ╔═╡ d4769c7b-22c6-4a2b-8943-ed0e4b11b749
reshape(D_10 * ptr(rho_t_inter,dim_1,2,order)[:],dim_1,dim_1)

# ╔═╡ 8c867b56-7980-4923-bc48-b6941238a555
reshape(D_inv * ptr(rho_t_inter,dim_1,2,order)[:],dim_1,dim_1)

# ╔═╡ 5d9ad62b-7c33-4da6-bfa7-5590e29b5c2e
eigen(reshape(D_inv * density_matrix(dim_1)[:],dim_1,dim_1))

# ╔═╡ 1edadbdf-3ed4-479c-94cf-61785b221d94
# Function to calculate the derivatives of c_ij using a function

function c_coefficients(t, start_rho = rho_0)
	D =  dynamical_map(t, start_rho)

	ew, ev = eigen(Hermitian(choi(D,2)))
	ew[abs.(ew) .< 10^-10] .= 0

	K = [reshape(ev[:,k],2,2) for k = 1:4] .* sqrt.(ew)
	
	K = K[ew .> 10^-13]
	
	#test if Kraus operators are correct
	#norm(D - sum([kron(conj.(K[k]), K[k]) for k = 1:numel_K]))


	
	d = [tr(A[i]' * K[j]) for i = 1:4, j = 1:length(K)]
	c = d*d'
	c[1,1] -=2
	return c
end

# ╔═╡ 1eaa2d93-b338-446f-9de0-e4516bb9963b
 begin 
	 c_coefficients(time) # returns the coefficients needed to derive Gamma matrix from the dynamical map Λ(t,t_0, rho_0)

 end

# ╔═╡ 2aef12dd-3095-4124-b2e3-c0faf4f5374d
begin
	time_Lindblad = 1.001
	
	#dynamical_map
	a = central_fdm(5,1)(c_coefficients, time_Lindblad)
	F = 1/4 * a[1,1] * I(2) + 1/sqrt(2) * sum([a[1,i] * A[i]' for i=2:4])
	G = (F+F')/2
	H_ = (F-F')/2im
	#test: should be zero!
	#norm(G-1/2* sum([a[i,j] * A[j]' * A[i] for i = 2:4, j = 2:4]))
	
	Γ = a[2:end,2:end] 
	
	#turns Lindblad form into superoperator (to make it compareable)
	L = -im * (kron(I(2),H_) - kron(transpose(H_),I(2))) + sum([a[i,j] * (kron(conj.(A[j]), A[i]) - 1/2*(kron(I(2), A[j]'*A[i]) + kron(transpose(A[j]'*A[i]), I(2)))) for i=2:4,j=2:4])

	#direct derivative of dynamical map:
	
	D_Λ = round.(central_fdm(5,1)(dynamical_map,time_Lindblad), digits=10)

	#test whether they are equal
	norm(D_Λ - L)
end

# ╔═╡ eeb71c28-e196-42d6-b31c-5efa5f84f432
round.(D_Λ, digits=5)

# ╔═╡ 5a5baf17-e71d-40f1-acdc-755f808e2c8f
round.(L, digits = 5)

# ╔═╡ 3c878be8-e26a-46ec-954b-059046f87b27
Γ

# ╔═╡ 53758ee5-3077-4d7c-ae5a-d939baeb4615
eigen(a[2:4,2:4])

# ╔═╡ 274b9ca1-8468-428f-8174-3f9946e0ddf2
eigen(a)

# ╔═╡ 89052766-79c4-4b4b-8cec-88e097bd7ab6
D_Λ

# ╔═╡ bb7bbeba-925a-46c2-8622-8f198255dd27
c_test = c_coefficients(time_Lindblad)

# ╔═╡ d9e5042a-24e1-4c1b-bd7d-1d3fd6e7dd90
c_test[1,1] +=2

# ╔═╡ d0d1e37e-3a76-4248-8744-b90cba30dd46
eigen(c_test[2:4, 2:4])

# ╔═╡ f294e282-36f3-4ff4-a5e1-590331103084
begin

K_der, eps_der = choi2kraus(choi(D_Λ,2), true)

d_der = [tr(A[i]' * K_der[k]) for i = 1:4, k=1:4]
a_der = d_der * Diagonal(eps_der) * d_der'

	
eigen(a_der[2:4, 2:4])
# check if the representation is correct
norm(D_Λ - sum([a_der[i,j] * kron(transpose(A[j]'), A[i]) for i=1:4, j=1:4]))


#NOTE: since it is basically not possible to find a CPTP map that is CP around t_0 = 0 starting with correlations, there exists no Lindblad form
end

# ╔═╡ 88664112-e1c1-4c5c-86a2-6f53d26e3f56
round.(a_der,digits=5)


# ╔═╡ 398b2f8b-6df4-4618-94f2-502e7e5787f4
norm(a-a_der)

# ╔═╡ 6e24af59-2efe-4568-85b0-81a23414f505
eigvals(choi(dynamical_map(time_Lindblad),2))

# ╔═╡ fa33473c-f341-4a7c-b5a6-a27a45d48a36
begin
TT = density_matrix(2)
	# do d/dt and Choi commute?
ddt_Choi = central_fdm(5,1)(t -> choi(exp(-im * (TT ⊗ I(2)) * t),2), 0)
Choi_ddt = choi(central_fdm(5,1)(t -> exp(-im * (TT ⊗  I(2)) * t), 0), 2)

norm(ddt_Choi - Choi_ddt)
	# anser is yes :D
end

# ╔═╡ 4fb4c614-ed74-411e-91db-641c331c75fa
eigvals(choi(Λ + Λ_c,2))
# map is not completely positive

# ╔═╡ 36df576e-5111-4d66-9170-2eb9573cea32
pos_test(choi(Λ + Λ_c,2),1,2)
# dynamical map is not positive

# ╔═╡ 5695cdc2-0da6-4ef1-9aa9-d553f9b194e4
begin
	
	
K_p = [1/2*(I(2) + σ[3]), 1/2*(I(2) - σ[3]), 1/sqrt(2)*σ[1]]
K_m = 1/sqrt(2)*[im * σ[2]]
Λ_t = sum([kron(conj.(K_p[i]), K_p[i]) for i = 1:3]) - kron(conj.(K_m[1]), K_m[1])
	
choi(Λ_t,2)	
	
C_t = sum([K_p[i][:] * K_p[i][:]' for i = 1:3]) - K_m[1][:] * K_m[1][:]'
	

C_t - kron(I(2), density_pauli([0,a1,a2,a3]))
end

# ╔═╡ d18f0a0f-53b1-48c1-9910-9a0454db6957
# correlation term

function correlation_term(time, rho = rho_0)
	rho_time = exp(-im * H * time) * rho * exp(im * H * time)
	rho_tens, rho_c = factorize(rho_time,dim_1,2,order)
	return rho_c
end
	

# ╔═╡ e1abbd34-a276-4a79-939d-45a8a2af118c
begin
plot(t_time, eig_vector', title = "Eigenvalues of Choi matrix", label = ["λ₁" "λ₂" "λ₃"  "λ₄"])
ew_D = eigvals.(D)
	
plot!(t_time, round.(sum.([abs.(ew_D[i]) for i = 1:length(ew_D)]), digits = 5), label = "unitary check", color = :blue)
plot!(t_time, real.(round.(sum.((ew_D)), digits = 5)), label = "trace", color=:blue, linestyle = :dash)
plot!(t_time, real.(tr.(sqrt.(D.*transpose.(conj.(D))))), color = :blue, linestyle = :dot, label = "sum of sing. values")
plot!(t_inter_time, eig_vector_inter', linestyle = :dash, ylim = [-0.3,8.01], label = ["λ₁" "λ₂" "λ₃"  "λ₄"])
#subplot = twinx()
plot!(t_inter_time, 0.5 .* (-minimum(eig_vector_inter,dims=1) .> 10^-10)', color = :red, label = "violates cp?", legend = :outerright, rightmargin = -50mm)
plot!(t_time,4 .+ norm.(correlation_term.(t_time)), color = :magenta, label = "|ρ_c|", ylim = [-.01, 4.51])
	
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
Kronecker = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
FiniteDifferences = "~0.12.18"
Kronecker = "~0.5.0"
Plots = "~1.22.3"
PlutoUI = "~0.7.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "bc3930158d2be029e90b7c40d1371c4f54fa04db"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.6"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "StaticArrays"]
git-tree-sha1 = "9a586f04a21e6945f4cbee0d0fb6aebd7b86aa8f"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.18"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "f6e3cc35572a6be64308ffb0a56d70be36dc6a85"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "1bb9558fad77d915edd65ef84772a6cd91214346"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.41"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "cfbd033def161db9494f86c5d18fbf874e09e514"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.3"

[[PlutoUI]]
deps = ["Base64", "Dates", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "d1fb76655a95bf6ea4348d7197b22e889a4375f4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.14"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e03ca566bec93f8a3aeb059c8ef102f268a38949"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─21712488-f636-4861-913c-012693cb524d
# ╠═89b58dd0-c70c-4566-b991-2d739990f2ff
# ╟─23c019dd-53d2-461e-a6ba-f1b5da2f6710
# ╠═f5992b42-ff99-479a-b093-70a9bfd85655
# ╠═0a95d032-8576-492f-bee1-ef67303bcb3a
# ╟─85bca31d-c3be-46b6-b330-b80648472378
# ╟─ebd5efea-94b6-419e-8ff1-9dcf624ddaba
# ╟─c4406548-2119-480f-9438-22cd8c27e8f3
# ╠═aedb8227-6f24-43ae-9634-32cf365ccdcc
# ╠═c8e02ee8-2c21-4a37-9875-c4a463e0b462
# ╟─913d405a-e525-49d6-9021-227543b37953
# ╟─4f87569b-777e-4e97-ac80-8b7b76115d2a
# ╠═a36e5515-e7f9-4c34-a48e-52a004d80f23
# ╠═a5e38ca3-8184-4160-a6b1-4c1783c6e185
# ╟─5afc42f7-3253-438d-96a8-d665a1b9a1cd
# ╟─fc12024b-2785-4f89-a457-95a2deb5045f
# ╠═62b3f7c0-5b11-4984-b433-55b6bd40b8a7
# ╠═e1abbd34-a276-4a79-939d-45a8a2af118c
# ╟─f850340c-7a18-4693-8822-7ef29e3d2675
# ╠═3b7400dd-1c04-47f6-9f65-52bdedb63002
# ╠═580ff266-927a-447a-9a48-9e64bf07b806
# ╠═00ed1878-0fc5-464e-8f9c-66484f4f369e
# ╟─40b50286-d099-41a1-b73c-167625b0221c
# ╠═994c73ac-d001-4930-b28a-705ea98cada7
# ╠═596a9128-2fff-47e8-8b6f-e9fd8686a279
# ╠═b595349b-b572-4cc8-92b1-07631cc8fad7
# ╠═c578581e-467e-482c-9cce-2ad55d32267f
# ╟─7479e663-e533-442c-8324-c60f4849c470
# ╟─0872a46c-4f4b-49fa-8152-9193b71f332b
# ╠═8c779399-7db0-421e-b1b2-659d28fc8c41
# ╠═9a7de167-8474-4a22-99bf-66af8bffa736
# ╠═872b503b-4c40-4944-aefb-e52c559c72d5
# ╠═98c23700-c267-4c40-9a45-f5bb382c8429
# ╠═95c1685e-039b-492b-8a8c-d2914036addb
# ╟─c39065df-e7dc-4f09-b5c5-6066eec391ba
# ╠═a3ace2bc-5263-4868-a999-098f70e19483
# ╠═3006719e-ed10-4186-ade1-7c474af59883
# ╠═a3c9bf88-4404-4a6c-9c18-f2e58c325729
# ╠═10a8a0e7-a7e0-48ba-a177-6df6cbeb5f78
# ╠═9863549d-e86e-4b0c-9c18-804b233cea89
# ╠═7835a031-cffe-4d47-a47c-d8631f98ef2a
# ╠═254be7e0-d60a-45dc-8113-46fffe2cbd88
# ╠═88dcaaef-4782-4c64-acbe-91c9052cc955
# ╠═b4e4baa5-338a-43b4-9035-d6be8e2d5e17
# ╟─4dae271b-27b9-460e-988c-d733d8d79e87
# ╠═2921da97-0099-4662-9d7e-4a8696d85de5
# ╟─19b181eb-8c95-41e2-9f4a-786cb5a25b8b
# ╠═38459aa7-557c-4252-a150-ba475a52a860
# ╠═001db95a-4b60-483f-a863-7142269c9bbf
# ╠═455440b2-fc42-4db3-b9dd-eedd3406103a
# ╠═67c8115a-eb59-4215-a23d-a00065baeaba
# ╟─43b7be4e-9bae-4b1e-b536-4dd13aee7712
# ╟─4754c50a-0d88-477d-a832-1c0eb8ccddac
# ╠═908235d9-f919-4d89-a9d8-d172f0bce524
# ╠═13d983b3-d9af-4360-81c9-2fbc91907a2a
# ╟─b7dfd6ac-6246-4655-a2dd-3d0e46e2b87e
# ╠═e20411a0-c43f-44c5-8a60-252348ef9501
# ╠═c2af9765-647f-4f25-adc9-139f7333413c
# ╠═4c85b70f-7d65-449e-9c17-9db3414e918a
# ╠═48677999-941a-437c-90c2-e35ef44ee5f0
# ╠═49b3aeaf-ff2b-4805-bf4b-58aed5c49607
# ╠═50eca964-09a9-48ed-9b75-d3ac83e8b1d7
# ╠═17264c91-b88a-4e92-a336-3695e4263037
# ╠═db14e9c5-2b72-4f8d-b1bc-c2f9d11b67a1
# ╠═f80055ca-bea7-4ef3-80c2-a8892b374df3
# ╠═8393a842-397c-4600-bbca-2345ea06722e
# ╠═e8c989d6-9ba6-40d1-ad36-d768172d5f5e
# ╠═9330d1cf-e0e1-41c1-a876-5fae11633ce8
# ╠═4f8ada04-10a7-4783-8709-db55428824b3
# ╠═5d450f8d-c8c4-4a58-b2a6-97bfe109f91f
# ╟─9a519b1a-6b0f-430c-869a-f8b902e862b6
# ╠═ea84b314-dfaf-4799-ac46-541b6a1318cd
# ╠═f617e5dc-f2f7-4d79-8258-b3c95de86653
# ╠═a2747d76-9f86-4bad-87a7-e71b9a24cbe8
# ╠═3239e943-5328-41fd-9435-798504eb34a8
# ╠═7179873b-53d2-44d4-a618-63ca58293b8d
# ╠═637a2af1-d61a-4dff-aa30-f45c4ac95f6a
# ╠═59ee6550-c3ec-4874-8afe-e852a25d15b6
# ╠═f7915998-5d2b-4fcb-bccf-c2163d9dc71b
# ╠═79a90d41-8846-49ac-8dc5-e615a9c26e38
# ╠═1044ff8d-3778-4f69-93ec-fe4335cba189
# ╠═eae36bd4-9733-43b1-8375-f1b3eab87aa0
# ╠═d6619751-decf-40a4-874a-cbf0571de81c
# ╟─be1e32d7-a48a-404c-b25c-e435a9b4c0ce
# ╠═b19b685a-edb2-4913-b21f-68015b961911
# ╠═7e319790-26ac-4409-8d5d-ebc2962a87da
# ╠═2cd1627f-6056-4c36-b089-c563097853f9
# ╠═66b53661-7f06-4290-94a1-28240e29f067
# ╠═9e17a64b-4b49-44fb-8b5d-b409d6897de1
# ╠═a8c70b91-6783-4507-9194-0a26f1e49860
# ╠═1eaa2d93-b338-446f-9de0-e4516bb9963b
# ╟─6d2b167f-ded7-4e6d-9d77-e2a555d1b46c
# ╠═c3696c66-4b84-40b6-af9f-d3c22ea84e35
# ╟─80f1a784-acf1-4df0-a36b-02665257143f
# ╟─8f273160-c0ab-41e2-85ac-653db2ee7438
# ╠═7a6ea79a-8c86-4bd7-ab36-92b9ccbe2289
# ╟─b9bc1429-bd9a-452b-9ddf-34b91217de96
# ╠═08606417-ef27-4944-afe1-42cdba38e2e4
# ╠═cc05e609-805a-47de-8824-dc446d2676e2
# ╠═06883a55-b99c-4edd-ae8b-e9cc9abe2b99
# ╠═f29ab0fa-72f6-49a4-abc2-492a32a88374
# ╠═754b5aaf-841c-445d-ad8f-06a8e7ed391d
# ╠═d170f96b-f329-482e-be03-0cbfc75c41dc
# ╠═038aaf32-fccc-45b3-a586-248ab7b7e45f
# ╠═d4769c7b-22c6-4a2b-8943-ed0e4b11b749
# ╠═15cb7806-8b8a-4920-9197-18dd80a67926
# ╠═884c70c1-defe-4680-a575-6915a2917abc
# ╠═8c867b56-7980-4923-bc48-b6941238a555
# ╠═ffa20581-01a4-44c4-b89e-a4732243522c
# ╠═5d9ad62b-7c33-4da6-bfa7-5590e29b5c2e
# ╠═5d00c2ec-77c9-49a6-8020-a1919ce99bbb
# ╠═6a2eecf5-998f-44b2-8576-db0092e706d9
# ╠═2aef12dd-3095-4124-b2e3-c0faf4f5374d
# ╠═eeb71c28-e196-42d6-b31c-5efa5f84f432
# ╠═5a5baf17-e71d-40f1-acdc-755f808e2c8f
# ╠═1edadbdf-3ed4-479c-94cf-61785b221d94
# ╠═3c878be8-e26a-46ec-954b-059046f87b27
# ╠═3553ea01-ba81-48cd-be9f-886508a70a2d
# ╠═f294e282-36f3-4ff4-a5e1-590331103084
# ╠═eaafb3a3-a12b-4c5a-947a-2bcd376612c0
# ╠═88664112-e1c1-4c5c-86a2-6f53d26e3f56
# ╠═53758ee5-3077-4d7c-ae5a-d939baeb4615
# ╠═bb7bbeba-925a-46c2-8622-8f198255dd27
# ╠═d9e5042a-24e1-4c1b-bd7d-1d3fd6e7dd90
# ╠═d0d1e37e-3a76-4248-8744-b90cba30dd46
# ╠═274b9ca1-8468-428f-8174-3f9946e0ddf2
# ╠═398b2f8b-6df4-4618-94f2-502e7e5787f4
# ╠═89052766-79c4-4b4b-8cec-88e097bd7ab6
# ╠═6e24af59-2efe-4568-85b0-81a23414f505
# ╟─022cc2fa-f6d4-4fec-b7ac-645890984d46
# ╠═fa33473c-f341-4a7c-b5a6-a27a45d48a36
# ╟─558e2f3a-2942-4062-982e-f5c94c0f9588
# ╠═878a884d-196c-4403-bc86-9e642956441c
# ╟─5e1377b7-c784-4e42-8190-ffbf48f0ecc1
# ╟─822ea5ab-4a3a-45ee-903c-ff9bad9d5f16
# ╠═4fb4c614-ed74-411e-91db-641c331c75fa
# ╠═8cd0438c-c480-4b73-93af-79e3f3a5dfa9
# ╠═36df576e-5111-4d66-9170-2eb9573cea32
# ╠═1e4adc4d-52cf-41df-ad62-cab016f55397
# ╠═2cab05f7-bc59-4d26-a535-495b5ab104c0
# ╠═faf456bc-9145-4d87-ad4d-6ecc8c78bd41
# ╠═e63d17ba-767c-4136-8dc3-3da609f8dda5
# ╠═002504f4-e74a-4e8c-80fa-f4c5f2e90df0
# ╠═1d31452b-5ccf-49ba-ac94-d5c28febe295
# ╟─4cdab1a4-1784-4430-ae4f-50a4459d83a4
# ╠═5695cdc2-0da6-4ef1-9aa9-d553f9b194e4
# ╠═0c09a3b6-920d-4732-af67-86b36bd88292
# ╟─a558b847-3f09-473c-9ec5-70badcda7589
# ╟─8407fc8c-95b4-4536-bbda-b3f4945518b0
# ╟─104b6690-212b-11ec-3a2b-25c420f62534
# ╠═7f43573f-010f-455a-82c7-f2f509cce3d3
# ╠═d28b8274-7435-484a-bde7-61443d7e4ae2
# ╠═dc95c17d-de0f-4b72-b57e-0dab473a69cd
# ╠═5663ed51-a921-4a8b-b351-86aca24420b6
# ╠═5c9ac0b7-11e7-4f52-90ff-04ce131c61d8
# ╠═75ebae9f-1c69-4d92-9cdf-a0f4b0971576
# ╠═d18f0a0f-53b1-48c1-9910-9a0454db6957
# ╟─3140729e-d516-434d-ba32-3ff9705ee83b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
