from matplotlib.pyplot import *


def apply_style(sty="vimuq1", use_case=""):

    # reset style
    print("I'm resetting rcParams to matplotlib default.")
    style.use("default")

    if sty == "vimuq1":
        print(f"I'm setting rcParams to style {sty}.")
        # Settings related to plots
        # fonts
        rcParams["text.usetex"] = True
        rcParams["mathtext.fontset"] = "cm"
        rcParams["font.serif"] = ["Computer Modern"]
        rcParams["font.family"] = "sans-serif"
        rcParams["font.size"] = 9

        # figure
        rcParams["figure.titlesize"] = "medium"
        rcParams["figure.figsize"] = [6.4, 4.8]  # in inches; [16.256, 12.192] cm
        rcParams["figure.dpi"] = 100  # use 300 if raster file format

        # legend
        rcParams["legend.columnspacing"] = 0.5
        rcParams["legend.markerscale"] = 0.2
        rcParams["legend.handlelength"] = 1.0
        rcParams["legend.handleheight"] = 0.35
        rcParams["legend.handletextpad"] = 0.4
        rcParams["legend.columnspacing"] = 0.25
        rcParams["legend.loc"] = "best"
        rcParams["legend.borderaxespad"] = 0.25
        rcParams["legend.fontsize"] = "small"
        rcParams["legend.title_fontsize"] = "small"

        rcParams["axes.titlesize"] = "medium"

        rcParams["lines.linewidth"] = 1
        rcParams["hatch.linewidth"] = 2

    if sty == "vimuq2":
        print(f"I'm setting rcParams to style {sty}.")
        # fonts = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
        rcParams["mathtext.fontset"] = "custom"
        rcParams["mathtext.it"] = "Arial:italic"
        rcParams["mathtext.rm"] = "Arial"

        rcParams["font.size"] = 9

        rcParams["figure.figsize"] = [6.4, 4.8]
        rcParams["figure.dpi"] = 300
        rcParams["figure.titlesize"] = "large"

        rcParams["axes.titlesize"] = "medium"

        rcParams["legend.fontsize"] = "small"
        rcParams["legend.title_fontsize"] = "small"

    if sty == "default":
        style.use("default")

    # scale fig width
    # thesis
    thesis = dict()
    thesis["text.width"] = cm_to_inches(14.6979)
    thesis["text.height"] = cm_to_inches(20.9370)
    # figure width if multiple sub-figures are placed in the same row
    thesis["fig.oneup"] = 1 * thesis["text.width"]
    thesis["fig.twoup"] = 0.48 * thesis["text.width"]
    thesis["fig.threeup"] = 0.32 * thesis["text.width"]
    thesis["fig.fourup"] = 0.24 * thesis["text.width"]

    # other
    plos = dict()
    plos["text.width"] = cm_to_inches(13.3331)

    active_aspect_ratio = rcParams["figure.figsize"][0] / rcParams["figure.figsize"][0]

    if use_case == "thesis":
        rcParams["figure.figsize"] = [
            thesis["text.width"],
            thesis["text.width"] / active_aspect_ratio,
        ]
    if use_case == "plos":
        rcParams["figure.figsize"] = [
            plos["text.width"],
            plos["text.width"] / active_aspect_ratio,
        ]


def check_rcparams_changes(rc_params):
    p = "parameter"
    v = "value"
    dv = "default"
    l1 = 35
    l2 = 20
    l = l1 * "-" + "|" + (l2 + 1) * "-" + "|" + (l2 + 1) * "-" + "|"
    print(f"Modified rcParams:\n\n{p.ljust(l1)}| {v.ljust(l2)}| {dv.ljust(l2)}|\n{l}")
    for rcp in rc_params.items():
        if rcp[1] != rcParamsDefault[rcp[0]]:
            print(
                f"{rcp[0].ljust(l1)}| {str(rcp[1]).ljust(l2)}| {str(rcParamsDefault[rcp[0]]).ljust(l2)}|"
            )


def cm_to_inches(a):
    return a * 0.393701
