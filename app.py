# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import html, ctx, dcc
import plotly.express as px
import itertools
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
# import scipy.stats as ss
# import dash_bootstrap_components as dbc
from flask import Flask


server = Flask(__name__)
app = dash.Dash(__name__, server=server)#, external_stylesheets=[dbc.themes.JOURNAL])

dataset_keys = pd.read_excel(
    "/home/tomsasani/bosch-af-multimer-app/InsR.xlsx",
    sheet_name="Key",
)

datasets = [k.rstrip("\t") for k in dataset_keys["Type of dataset"].unique()]
data_to_fields = (
    dataset_keys.groupby("Type of dataset")
    .agg(scores=("Score", lambda s: list(set(s))))
    .to_dict()["scores"]
)
data_to_fields = {k.rstrip("\t"): v for k, v in data_to_fields.items()}

data = pd.read_excel(
    "/home/tomsasani/bosch-af-multimer-app/InsR.xlsx",
    sheet_name="InsR",
    index_col=[0, 1],
    header=[0, 1],
)


res_corr = []
for da, db in itertools.combinations_with_replacement(datasets, r=2):
    da_fields, db_fields = data_to_fields[da], data_to_fields[db]
    da_data, db_data = data[da], data[db]
    for a_field, b_field in itertools.product(da_fields, db_fields):
        a_field_new, b_field_new = a_field, b_field
        if a_field_new == b_field_new:
            a_field_new += "_A"
            b_field_new += "_B"
        avals = da_data[a_field].rename(a_field_new)
        bvals = db_data[b_field].rename(b_field_new)
        merged = pd.concat([avals, bvals], axis=1).dropna()
        
        correlation = np.corrcoef(merged[a_field_new].values, merged[b_field_new].values)
        res_corr.append(
            {
                "Dataset A": da,
                "Field A": a_field,
                "Dataset B": db,
                "Field B": b_field,
                "Correlation": correlation,
                #"p": -1,
            }
        )
res_corr = pd.DataFrame(res_corr)


app.layout = (
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """
    ## Exploring AlphaFold-Multimer data

    This app allows users to visualize pairwise comparisons between various InsR datasets.
    """
            )
        ],
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """### Generate a scatterplot comparing scores in two datasets."""
            ),
            dcc.Markdown(
                """Select two datasets to compare. Then, select the specific field from each dataset you want to plot. If the first dataset is 'AlphaMissense', you can optionally plot AlphaMissense scores as `discrete` (benign, ambiguous, or pathogenic) or `continuous` (the raw values output by AlphaMissense). """
            ),
        ],
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """**Select dataset A:**""",
                style={"width": "15%"},
            ),
            dcc.Dropdown(
                id="a-data-dropdown",
                options=[{"label": i, "value": i} for i in datasets],
                value="AlphaMissense",
                placeholder="Select a dataset",
                style={"width": "35%"},
            ),
            dcc.Markdown(
                """**Select field from dataset A:**""",
                style={"width": "15%"},
            ),
            dcc.RadioItems(
                id="a-field-radio",
                style={"width": "20%"},
            ),
            dcc.RadioItems(
                id="discrete-bool",
                style={"width": "10%"},
            ),
        ],
        style={"display": "flex"},
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """**Select dataset B:**""",
                style={"width": "15%"},
            ),
            dcc.Dropdown(
                id="b-data-dropdown",
                options=[{"label": i, "value": i} for i in datasets],
                value="Aslanzadeh et al InsR saturation mutagenesis MAVE",
                placeholder="Select a dataset",
                style={"width": "35%"},
            ),
            dcc.Markdown(
                """**Select field from dataset B:**""",
                style={"width": "15%"},
            ),
            dcc.RadioItems(
                id="b-field-radio",
                style={"width": "20%"},
            ),
        ],
        style={"display": "flex"},
    ),
    html.Div(
        className="row",
        children=[
            dcc.Graph(
                id="graph-correlation",
                style={"width": "100%", "height": "100%"},
            ),
        ],
        style={"display": "flex"},
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """### How well do the AF-Multimer datasets recapitulate AlphaMissense pathogenicity scores?"""
            ),
            dcc.Markdown(
                """First, select a dataset to compare against AlphaMissense. 
                        Then, select a specific field from that dataset, which should contain continuous 'scores' for every possible missense change. 
                        We calculate the average score for missense changes predicted to be `benign` or `pathogenic` by AlphaMissense, and compute the difference
                        between the average `pathogenic` score and the average `benign` score. 
                        **If our scores are highly correlated with AlphaMissense pathogenicity, we'd expect the average `pathogenic`
                        score to be very different from the average `benign` score.**
                        We then permute the AlphaMissense labels (i.e., whether a missense change was predicted to be `pathogenic` or `benign`) 1,000 times, re-calculating the average score in the `benign` and `pathogenic` categories
                        each time. 
                        By comparing the 'true' averages to the 'shuffled' averages, we can determine whether the average `pathogenic` and `benign` scores
                        are more different than we'd expect by chance."""
            ),
            dcc.Markdown(
                """
                        **NOTE:** the vertical line represents the difference between the empirical `pathogenic` and `benign` means, while the histogram represents 1,000 permuted mean differences."""
            ),
        ],
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """**Select dataset:**""",
                style={"width": "10%"},
            ),
            dcc.Dropdown(
                options=[
                    {"label": i, "value": i} for i in datasets if i != "AlphaMissense"
                ],
                value="Aslanzadeh et al InsR saturation mutagenesis MAVE",
                id="perm-dataset-dropdown",
                style={"width": "50%"},
            ),
            dcc.Markdown(
                """**Select field from the dataset:**""",
                style={"width": "20%"},
            ),
            dcc.RadioItems(
                id="perm-field-radio",
                style={"width": "20%"},
            ),
        ],
        style={"display": "flex"},
    ),
    html.Div(
        className="row",
        children=[
            dcc.Graph(
                id="graph-permutation",
                style={"width": "100%", "height": "100%"},
            ),
        ],
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """### Visualize the Spearman's correlation between scores in two datasets."""
            ),
            dcc.Markdown(
                """Select two datasets to compare. Then, plot the Spearman's correlation coefficient between every pair of scores from those two datasets using either a `barplot` or `heatmap`."""
            ),
        ],
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """**Select dataset A:**""",
                style={"width": "20%"},
            ),
            dcc.Dropdown(
                options=[{"label": i, "value": i} for i in datasets],
                value="AlphaMissense",
                id="pw-dataset-a",
                style={"width": "50%"},
            ),
        ],
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown(
                """**Select dataset B:**""",
                style={"width": "20%"},
            ),
            dcc.Dropdown(
                options=[{"label": i, "value": i} for i in datasets],
                value="Aslanzadeh et al InsR saturation mutagenesis MAVE",
                id="pw-dataset-b",
                style={"width": "50%"},
            ),
        ],
    ),
    html.Div(
        className="row",
        children=[
            dcc.Markdown("""**Select chart type:**""", style={"width": "20%"}),
            dcc.RadioItems(
                options=["barplot", "heatmap"],
                value="barplot",
                id="pw-plot-type",
                style={"width": "30%"},
                inline=True,
            ),
        ],
    ),
    html.Div(
        dcc.Graph(id="pw-correlation", style={"width": "80%", "height": "80%"}),
    ),
)


@app.callback(
    [Output("a-field-radio", "options"), Output("a-field-radio", "value")],
    Input("a-data-dropdown", "value"),
)
def a_radio_options(dropdown_value):
    _radio_options = data_to_fields[dropdown_value]
    options = [{"label": x, "value": x} for x in _radio_options]
    value = _radio_options[0]
    return options, value


@app.callback(
    [Output("b-field-radio", "options"), Output("b-field-radio", "value")],
    Input("b-data-dropdown", "value"),
)
def b_radio_options(dropdown_value):
    _radio_options = data_to_fields[dropdown_value]
    options = [{"label": x, "value": x} for x in _radio_options]
    value = _radio_options[0]
    return options, value


@app.callback(
    [Output("perm-field-radio", "options"), Output("perm-field-radio", "value")],
    Input("perm-dataset-dropdown", "value"),
)
def perm_radio_options(dropdown_value):
    _radio_options = data_to_fields[dropdown_value]
    options = [{"label": x, "value": x} for x in _radio_options]
    value = _radio_options[0]
    return options, value


@app.callback(
    [Output("discrete-bool", "options"), Output("discrete-bool", "value")],
    Input("a-data-dropdown", "value"),
)
def radio_options(dropdown_value):
    if dropdown_value == "AlphaMissense":
        options = [{"label": x, "value": x} for x in ["continuous", "discrete"]]
        value = "continuous"
        return options, value
    else:
        options = [
            {"label": x, "value": x}
            for x in [
                "continuous",
            ]
        ]
        value = "continuous"
        return options, value


@app.callback(
    Output("graph-correlation", "figure"),
    [
        Input("a-data-dropdown", "value"),
        Input("a-field-radio", "value"),
        Input("b-data-dropdown", "value"),
        Input("b-field-radio", "value"),
        Input("discrete-bool", "value"),
    ],
)
def make_correlation_plot(a_name, a_score, b_name, b_score, discrete_value):

    if discrete_value == "discrete":
        if a_name == "AlphaMissense":
            a_score = "am_class"

    a_vals = data[a_name][a_score].to_frame()
    b_vals = data[b_name][b_score].to_frame()

    a_vals.rename(columns={a_score: "score_A"}, inplace=True)
    b_vals.rename(columns={b_score: "score_B"}, inplace=True)

    a_vals["dataset_name_A"] = a_name
    b_vals["dataset_name_B"] = b_name

    a_vals["score_name_A"] = a_score
    b_vals["score_name_B"] = b_score

    merged = a_vals.merge(b_vals, left_index=True, right_index=True)

    if discrete_value == "discrete":
        fig = px.box(
            merged, x="score_A", y="score_B", color="score_A", template="ggplot2"
        )
    else:
        fig = px.scatter(
            merged,
            x="score_A",
            y="score_B",
            # trendline="ols",
            template="ggplot2",
        )
        fig.update_traces(opacity=0.1)

    return fig


@app.callback(
    Output("graph-permutation", "figure"),
    [
        Input("perm-dataset-dropdown", "value"),
        Input("perm-field-radio", "value"),
    ],
)
def make_permutation_plot(perm_dataset, perm_score):

    a_vals = data["AlphaMissense"]["am_class"].to_frame()
    b_vals = data[perm_dataset][perm_score].to_frame()

    a_vals.rename(columns={"am_class": "AlphaMissense pathogenicity"}, inplace=True)

    merged = a_vals.merge(b_vals, left_index=True, right_index=True)

    means = (
        merged.groupby("AlphaMissense pathogenicity")
        .agg(mean=(perm_score, "mean"))
        .reset_index()
    )
    means["perm"] = 0
    means = means.pivot(
        index="perm", columns="AlphaMissense pathogenicity", values="mean"
    ).reset_index()
    means["diff"] = means["pathogenic"] - means["benign"]
    means["is_empirical"] = True

    res = [means]

    n_perms = 100

    for i in range(n_perms):
        merged["AlphaMissense pathogenicity"] = np.random.permutation(
            merged["AlphaMissense pathogenicity"]
        )
        means = (
            merged.groupby("AlphaMissense pathogenicity")
            .agg(mean=(perm_score, "mean"))
            .reset_index()
        )
        means["perm"] = i + 1
        means = means.pivot(
            index="perm", columns="AlphaMissense pathogenicity", values="mean"
        ).reset_index()
        means["diff"] = means["pathogenic"] - means["benign"]
        means["is_empirical"] = False
        res.append(means)
    res = pd.concat(res)

    perm = res[res["is_empirical"] == False]
    emp = res[res["is_empirical"] == True]["diff"].values[0]

    pval = np.sum(np.abs(perm["diff"]) >= np.abs(emp)) / n_perms

    fig = px.histogram(
        perm, x="diff", template="ggplot2", title=f"Empirical two-sided p-value: {pval}", labels={
                     "diff": "Difference between average pathogenic and benign scores",
                 },
    )
    fig.add_vline(x=emp, line_width=2, fillcolor="dodgerblue", name="boo")

    mmin, mmax = min([emp, perm["diff"].min()]), max([emp, perm["diff"].max()])
    fig.update_layout(
        xaxis=dict(range=[mmin * 1.05 if mmin < 0 else mmin * 0.95, mmax * 1.05]),
        showlegend=True,
    )
    return fig


@app.callback(
    Output("pw-correlation", "figure"),
    [
        Input("pw-dataset-a", "value"),
        Input("pw-dataset-b", "value"),
        Input("pw-plot-type", "value"),
    ],
)
def make_pairwise_plot(dataset_a, dataset_b, plot_type):

    res_corr_filtered = res_corr[
        (res_corr["Dataset A"] == dataset_a) & ((res_corr["Dataset B"] == dataset_b))
        | (res_corr["Dataset A"] == dataset_b) & ((res_corr["Dataset B"] == dataset_a))
    ]

    if plot_type == "heatmap":
        res_corr_filtered = res_corr_filtered[
            ["Field A", "Field B", "Correlation"]
        ].pivot(index="Field A", columns="Field B", values="Correlation")
        fig = px.imshow(res_corr_filtered, range_color=[-1, 1])
    elif plot_type == "barplot":
        fig = px.bar(
            data_frame=res_corr_filtered,
            x="Field A",
            y="Correlation",
            color="Field B",
            barmode="group",
            # hover_data=["p"],
            template="ggplot2",
            width=1200,
            height=600,
            range_y=[-1, 1],
        )

    return fig


if __name__ == "__main__":
    app.run()
