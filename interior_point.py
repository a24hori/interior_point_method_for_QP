# -*- coding: utf-8 -*-
'''interior_point.py

    "interior_point.py" is to solve the following formed convex
    quadratic programming problem by using the (primal-dual)
    interior point method:\
        min_x   x^T G x + q^T x + c
        s.t.    Ax≧b,
    where G is positive semidefinite and A is a m-row full rank matrix.

    Coded by Atsushi Hori (Kyoto Univ.), Oct. 8th, 2020.
'''

from numpy import *
import time

# User defined:
G = array([[1,-1],[-1,2]])
q = array([-2,-6])
const = 0
A = array([[-1/2,-1/2],[1,-2],[1,0],[0,1]])
b = array([-1,-2,0,0])
n = len(q)
m = len(A)

def objfun(x):
    return x@G@x + q.T@x + const

def gradobj(x):
    return G@x + q

def hessobj(x):
    return G

def constr(x):
    return A@x - b

def gradconstr(x):
    return A.T

def lagrangian(x, lmd):
    return objfun(x) + dot(lmd, constr(x))

def gradlag(x, lmd):
    return gradobj(x) + gradconstr(x)@lmd

def solveapprxKKT(xk,yk,lk,Yaff,Laff,sigma):
    Y = diag(yk)
    L = diag(lk)
    rd = G@xk + q - A.T@lk
    rp = A@xk - b - yk
    mu = yk@lk/m
    #print(block([G, zeros((n,m)), A.T]))
    #print(block([A, eye(m), zeros((m,m))]))
    #print(block([zeros((m,m)), L, Y]))
    JF = block([[G, zeros((n,m)), -A.T],
                [A, -eye(m), zeros((m,m))],
                [zeros((m,n)), L, Y]])
    #print(linalg.det(JF))
    F = block([-rd, -rp, -L@Y@ones(m)-Laff@Yaff@ones(m)+(sigma*mu)*ones(m)])
    #print('JF:\n{}'.format(JF.shape))
    #print('F:\n{}'.format(F.shape))
    sol = linalg.solve(JF, F)
    dx = sol[0:n]
    dy = sol[n:n+m]
    dl = sol[n+m:n+m+m]
    return [dx, dy, dl]

def solve():
    # Algorithm "Predictor-Corrector Algorithm for QP" from
    #   Algorithm 16.4 in Nocedal and Wright (p.484).
    
    maxiter = 5000     # maximum number of iterations
    epsilon = 1.0e-5   # tolerance for stopping criteria  
    k       = 0        # number of iterations

    # Initialization
    init_x = ones(n)
    init_y = 1*ones(m)
    init_l = 2*ones(m)

    [dxaff, dyaff, dlaff] = \
        solveapprxKKT(init_x,init_y,init_l,zeros((m,m)),zeros((m,m)),0)
    xk = init_x
    yk = maximum(ones(m), abs(init_y + dyaff))
    lk = maximum(ones(m), abs(init_l + dlaff))

    tau = 1e-2
    for k in range(maxiter):
        # Solve affine set problem
        [dxaff, dyaff, dlaff] = solveapprxKKT(xk,yk,lk,zeros((m,m)),zeros((m,m)),0)
        
        mu = yk@lk/m

        alpha_aff = 1
        while any(block([yk,lk])+alpha_aff*block([dyaff,dlaff]) < zeros(2*m)):
            alpha_aff = alpha_aff - 0.1

        mu_aff = (yk+alpha_aff*dyaff)@(lk+alpha_aff*dlaff)/m

        sigma = (mu_aff/mu)**3

        [dx, dy, dl] = solveapprxKKT(xk,yk,lk,diag(dyaff),diag(dlaff),sigma)

        tau = 1 - 0.5**(k+1)

        alpha_pri = 1
        alpha_dual = 1
        while any(yk+alpha_pri*dy < (1-tau)*yk):
            alpha_pri = alpha_pri - 0.1
        while any(lk+alpha_dual*dl < (1-tau)*lk):
            alpha_dual = alpha_dual - 0.1
        alpha = min(alpha_pri,alpha_dual)

        xk = xk + alpha*dx
        yk = yk + alpha*dy
        lk = lk + alpha*dl

        if any(abs(block([dx,dy,dl])) < epsilon*ones(n+m+m)):
            break

    return [xk, yk, lk]

def main():
    start_time = time.time()
    [xk, yk, lk] = solve()
    end_time = time.time()
    print('comput. time: {} sec.'.format(end_time - start_time))
    print('sol:\n{}'.format(xk))
    print('fval(sol): {}'.format(objfun(xk)))

    
if __name__ == '__main__':
    main()