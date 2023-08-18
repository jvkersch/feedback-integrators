from shiny import App, reactive, render, ui

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from symbolic import q, p, derived_C, X_full_feedback, to_function

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
            ui.input_slider("k", "Feedback gain", value=50, min=0, max=100),
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
        k = input.k()
        Xff = to_function(X_full_feedback(H, C1, C2, k, k, k, q0, p0))

        t_eval = np.linspace(0, 10, 100)
        result = solve_ivp(lambda t, y: Xff(y), [0, 10], q0 + p0, t_eval=t_eval)
        return result

    @output
    @render.text
    def vector_field():
        H = sp.sympify(input.H())
        C1 = sp.sympify(input.C1())
        C2 = derived_C(H, input.C1())
        k = input.k()

        X = X_full_feedback(H, C1, C2, k, k, k, q0, p0)
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

        fig, (ax_h, ax_c1, ax_c2) = plt.subplots(ncols=3)
        ax_h.plot(result.t, h_data)
        ax_c1.plot(result.t, c1_data)
        ax_c2.plot(result.t, c2_data)

        return fig

    @output
    @render.text
    def diagnostics():
        return f"Function evaluations: {run_solver().nfev}"


app = App(app_ui, server)
