import pytest
import sympy as sp

from symbolic import q, p, delta, dot, X_full, derived_C, to_function


def eval_at(expr, args):
    subs = dict(zip(q + p, args))
    return sp.simplify(expr).evalf(subs=subs)


def test_conservation():
    H = sp.sympify("(px^2 + py^2 + pz^2)/2 + (x^2 + y^2 + z^2)/2")
    C1 = sp.sympify("x^2 + y^2 + z^2 - 1")
    C2 = derived_C(H, C1)

    X = X_full(H, C1, C2)

    assert sp.simplify(dot(X, delta(H))) == 0
    assert sp.simplify(dot(X, delta(C1))) == 0

    # Checking the conservation of C2 requires using the constraints
    # themselves. I don't know how to do that, so we fall back on just
    # substituting a point on the constraint surface and verifying that we get
    # 0.
    pt = (0, sp.sqrt(2)/2, sp.sqrt(2)/2, 5, -3, 3)
    assert eval_at(dot(X, delta(C2)), pt) == 0


def test_to_fun():
    H = sp.sympify("(px^2 + py^2 + pz^2)/2 + (x^2 + y^2 + z^2)/2")
    C1 = sp.sympify("x^2 + y^2 + z^2 - 1")
    C2 = derived_C(H, C1)

    X = X_full(H, C1, C2)
    X_fun = to_function(X)

    pt = (0, 2**0.5/2, 2**0.5/2, 5, -3, 3)
    assert X_fun(pt) == pytest.approx(
        [5.0, -3.0, 3.0, 0.0, -30.40559, -30.40559])
