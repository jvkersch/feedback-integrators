from shiny import App, reactive, render, ui

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from symbolic import q, p, derived_C, X_full_feedback, to_function

METHODS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]

# TODO hardcoded
q0 = (0, 2**0.5/2, 2**0.5/2)
p0 = (1, -3, 3)


app_ui = ui.page_fluid(
    ui.panel_title(
        ui.h2("Feedback integrator playground"),
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_text("H", "Hamiltonian", value="(px**2 + py**2 + pz**2)/2"),
            ui.input_text("C1", "Constraint", value="x**2 + y**2 + z**2 - 1"),
            ui.input_slider("k1", "Feedback gain (Energy)", value=50, min=0, max=100),
            ui.input_slider("k2", "Feedback gain (Constraint 1)", value=50, min=0, max=100),
            ui.input_slider("k3", "Feedback gain (Constraint 2)", value=50, min=0, max=100),
            ui.input_select("method", "Integration method", METHODS),
        ),
        ui.panel_main(
            ui.navset_tab(
                ui.nav(
                    "Plots",
                    ui.output_plot("conservation_laws_plot"),
                ),
                ui.nav(
                    "Vector fields",
                    ui.output_text("vector_field"),
                ),
                ui.nav(
                    "Solver diagnostics",
                    ui.output_text("diagnostics"),   
                ),
            ),
        ),
    ),
)


def server(input, output, session):

    @reactive.Calc
    def run_solver():
        H = sp.sympify(input.H())
        C1 = sp.sympify(input.C1())
        C2 = derived_C(H, input.C1())
        Xff = to_function(X_full_feedback(H, C1, C2, input.k1(), input.k2(), input.k3(), q0, p0))

        t_eval = np.linspace(0, 10, 100)
        result = solve_ivp(lambda t, y: Xff(y), [0, 10], q0 + p0, t_eval=t_eval, method=input.method())
        return result

    @output
    @render.text
    def vector_field():
        H = sp.sympify(input.H())
        C1 = sp.sympify(input.C1())
        C2 = derived_C(H, input.C1())

        X = X_full_feedback(H, C1, C2, input.k1(), input.k2(), input.k3(), q0, p0)
        return f"{X}"

    @output
    @render.plot
    def conservation_laws_plot():
        result = run_solver()

        h_fun = sp.lambdify([q + p], input.H())
        C1_fun = sp.lambdify([q + p], input.C1())

        C2 = derived_C(input.H(), input.C1())
        C2_fun = sp.lambdify([q + p], C2)
        h_data = np.array([h_fun(pt) for pt in result.y.T])
        h_data -= h_data[0]

        c1_data = np.array([C1_fun(pt) for pt in result.y.T])
        c2_data = np.array([C2_fun(pt) for pt in result.y.T])

        fig, axes = plt.subplot_mosaic([["A", "A", "A"],
                                        ["B", "C", "D"]])

        axes["A"].plot(result.t, result.y[0])
        axes["A"].plot(result.t, result.y[1])
        axes["A"].plot(result.t, result.y[2])
        axes["A"].set_title("Trajectory")
        axes["B"].plot(result.t, h_data)
        axes["B"].set_title("Energy Error")
        axes["C"].plot(result.t, c1_data)
        axes["C"].set_title("Constraint 1 Error")
        axes["D"].plot(result.t, c2_data)
        axes["D"].set_title("Constraint 2 Error")

        return fig

    @output
    @render.text
    def diagnostics():
        return f"Function evaluations: {run_solver().nfev}"


app = App(app_ui, server)
