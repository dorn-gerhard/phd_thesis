using Base: Bool
using LinearAlgebra
using SparseArrays

dim_1 = 2
dim_2 = 3

function unity(i,n)
    E = spzeros(Bool,n,n)
    E[i] = true
    return E
end

# xs = [rand(3,4) for _ in 1:5]
# reduce((x,y) -> cat(x,y,dims=3), A)
# idea: https://github.com/JuliaLang/julia/pull/37196
# reduce(cat(dims=4), [ones(2,3) for _ in 1:5]) |> size 
#`reduce(vcat, [rand(2,3) for _ = 1:4]): A[i][a,b] -> A[(i-1)*len_a + a, b]`
#which is the same as restructuring to` B[i,a][b] via B = [A[i][a,:] for i = 1:dim_i, a = 1:dim_a] `
#and then taking `reduce(hcat,B'[:]')'`
#`reduce(hcat, [rand(2,3) for _ = 1:4]): A[i][a,b] -> A[a, (i-1)*len_b + b]` 

function partial_trace(dim_1, dim_2, traced_system)

    if traced_system == 1
        T = [kron(sparse(I, dim_1,dim_1), unity(i,dim_2))[:] for i = 1:dim_2^2]
        col = reduce(vcat,[findnz(T[i])[1] for i = 1:dim_2^2])
        row = kron(1:dim_2^2, ones(dim_1))
        return sparse(row, col, trues(18))
    else
        return    Tr_B = [reshape(kron(reshape(unity(i,dim_1), dim_1, dim_1), I(dim_2)), 1, dim_2^2 * dim_1^2)[1,j] for i = 1:dim_1^2, j =1:dim_1^2*dim_2^2]
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

m = 7
n = 3

A = rand(m,m)
B = rand(n,n)


C = kron(A,B)

A_sup = kron(A[:], I(n^2))
C_sup = choi(C,m)

sum(abs.(A_sup * B[:] - C_sup[:]))

A_sup * B[:] - C[:]
reshape(A_sup * B[:],n*m,n*m) - C_sup
sum(abs.(ans[:]))


# create superoperator without applying Choi on the right hand side

G = reshape(permutedims(reshape(kron(A[:], I(n^2)), (n,n,m,m,n^2)), (1,3,2,4,5)), n^2*m^2,n^2)

reshape(G*B[:], m*n, m*n)
C
sum(abs.(G * B[:] - C[:]))

kron_fac_B = reshape(permutedims(reshape(kron(I(m^2), B[:]), (n,n,m,m,m^2)), (1,3,2,4,5)), n^2*m^2, m^2)
norm(reshape(kron_fac_B * A[:], n*m, n*m) - C)


function I_rect(n1,n2)
    if n1 >= n2
        temp = I(n1)
        temp = temp[:,1:n2]
    else
        temp = I(n2)
        temp = temp[1:n1,:]
    end
    return temp
end


m1 = 3
m2 = 4

n1 = 5
n2 = 6

A = rand(m1,m2)
B = rand(n1,n2)

C = kron(A,B)

#G = reshape(permutedims(reshape(kron(A[:], I(n1*n2)), (n1,n2,m1,m2,n1*n2)), (1,3,2,4,5)), n1*n2*m1*m2,n1*n2)

S_left = reshape(permutedims(reshape(kron(I(n1*n2), A[:]), (m1,m2,n1,n2,n1*n2)), (3,1,4,2,5)), prod([m1,m2,n1,n2]), n1*n2)
norm(S_left * B[:] - C[:])

S_right = reshape(permutedims(reshape(kron(B[:], I(m1*m2)), (m1,m2,n1,n2,m1*m2)), (3,1,4,2,5)), prod([m1,m2,n1,n2]), m1*m2)
norm(S_right * A[:] - C[:])


norm(kron(A[:], I(n1*n2)) * B[:] - choi_rect(C, n1,m1, n2, m2)[:])

norm(kron(I(m1*m2), B[:]) * A[:] - choi_rect(C,n1,m1,n2,m2)[:])

norm(kron(A'[:], I(n1*n2)) * B'[:] - C'[:])

# elegant version to show right hand side as a Choi matrix for first and second factor:
norm(kron(I(n1*n2) ,A[:]) * B[:] - choi_rect(C, n1, m1, n2, m2)'[:])
norm(kron(B[:], I(m1*m2)) * A[:] - choi_rect(C, n1, m1, n2, m2)'[:])


# check embeddings

A = rand(2)
B = rand(3)
C = rand(4)
D = rand(5)

Q = kron(A, B')
kron(kron(A, B'), ones(4,5))

kron(kron(A,B'), kron(C,D')) - kron(kron(A, B'), I(4)) * kron(I(4), kron(C,D'))

Q = kron(A, C')
kron(kron(A, ones(3)'), kron(C, ones(5)'))- kron(Q'[:], ones(3*5)')

Q = kron(A, D')
kron(kron(A, ones(3)'), kron(ones(4), D'))-kron(kron(ones(3)', Q), ones(4))

Q = kron(B, C')
kron(kron(ones(2), B'), kron(C, ones(5)')) - kron(kron(ones(2),Q'), ones(5)')

Q = kron(B, D')
kron(kron(ones(2), B'), kron(ones(4), D'))-kron(ones(2*4), Q[:]')

# multiplication table for nxn unity base matrices

n = 4
A = zeros(Int64,n^2,n^2)
function unity(i,n)
    E = spzeros(Bool,n,n)
    E[i] = true
    return E
end

for i = 1:n^2, j=1:n^2
    L = findall((unity(i,n)*unity(j,n))[:] .> 0)
    if isempty(L)
        A[i,j] = 0
    else
        A[i,j] = L[1]
    end
end
A



### density matrix

function density_matrix(n, neg = 0)
    #trace 1 remains preserved!
    dia = rand(n)
    if neg != 0
        dia[rand(1:n, neg)] .*= -1
    end
    dia = dia/sum(dia)
    ew, ev = eigen(rand(n,n))
    d_mat = ev* Diagonal(dia) * ev'
    return d_mat
end

# try to check \sum_{i != j} a_ii a_jj - abs(a_ij)^2 == sum_{i \neq j} \lambda_i lambda_j

A = density_matrix(10,0);
#Diagonal(real(A))
ew,ev = eigen(A);
L1 = [ A[i,i]*A[j,j] - abs(A[i,j])^2 for i = 1:5, j = 1:5];
L2 = [ew[i] * ew[j] * (i != j) for i = 1:5, j=1:5];

sum(real(L1) .< 0)
real(ew)'


# can every density matrix be represented as the sum of tensor products of valid density matrices?

A = density_matrix(4)
B = density_matrix(3)

ew, ev = eigen(kron(A,B))
sum(ew .< 0)

