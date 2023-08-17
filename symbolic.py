"""Symbolic computations with functions of (q, p)."""

import sympy as sp

q = x, y, z = sp.symbols("x, y, z")
p = px, py, pz = sp.symbols("px, py, pz")

# Primitives

def delta_q(f):
    return [sp.diff(f, v) for v in q]


def delta_p(f):
    return [sp.diff(f, v) for v in p]


def delta(f):
    return delta_q(f) + delta_p(f)


def dot(v1, v2):
    return sum(v*w for (v, w) in zip(v1, v2))


def plus(v1, v2):
    return [v+w for (v, w) in zip(v1, v2)]


def poisson(f1, f2):
    return dot(delta_q(f1), delta_p(f2)) - dot(delta_q(f2), delta_p(f1))


# Mechanics

def derived_C(H, C1):
    C2 = dot(delta_q(C1), delta_p(H))
    return C2


def X(f):
    X_q = [poisson(v, f) for v in q]
    X_p = [poisson(v, f) for v in p]
    return X_q + X_p


def X_constraint(H, C1, C2):
    C12 = 1/poisson(C1, C2)
    H_C1 = poisson(H, C1)
    H_C2 = poisson(H, C2)

    X_q = [-C12 * (-H_C1 * poisson(v, C2) + H_C2 * poisson(v, C1)) for v in q]
    X_p = [-C12 * (-H_C1 * poisson(v, C2) + H_C2 * poisson(v, C1)) for v in p]
    return X_q + X_p


def X_full(H, C1, C2):
    X_h = X(H)
    X_c = X_constraint(H, C1, C2)
    return plus(X_h, X_c)


# Utilities

def to_function(X):
    callables = [sp.lambdify([q + p], v) for v in X]

    def fun(args):
        return [c(args) for c in callables]

    return fun
