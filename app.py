from shiny import App, render, ui

import sympy as sp





app_ui = ui.page_fluid(
    ui.h2("Feedback integrator playground"),
    ui.input_text("hfun", "Hamiltonian", value="p**2/2 + q**2/2"),
    ui.input_text("c1fun", "Constraint", value="q**2 - 1"),
    ui.input_slider("h", "Step size", value=0.1, min=1e-3, max=1.0),
    ui.input_slider("K", "Feedback gain", value=5, min=0.1, max=10),
    ui.output_text_verbatim("debug"),
)

def server(input, output, session):
    @output
    @render.text
    def debug():
        h_fun = sp.sympify(input.hfun())
        c1_fun = sp.sympify(input.c1fun())

        dh_dq = sp.diff(h_fun, q)
        dh_dp = sp.diff(h_fun, p)
        dc1_dq = sp.diff(c1_fun, q)
        dc1_dp = sp.diff(c1_fun, p)

        c2_fun = dc1_dq*dh_dp

        c1_c2_bracket = _poisson(c1_fun, c2_fun)
        return f"{_poisson(q, h_fun)}     {_poisson(p, h_fun)}"


app = App(app_ui, server)
