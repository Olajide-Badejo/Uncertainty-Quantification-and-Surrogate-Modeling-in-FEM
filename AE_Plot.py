import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# Typography (LaTeX-like)
# -----------------------------
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["mathtext.fontset"] = "stix"

# -----------------------------
# Table content
# -----------------------------
left_col = [
    r"AE force latent dim",
    r"AE learning rate",
    r"AE max epochs",
    r"GPR kernel",
    r"GPR noise ($\alpha$)"
]

left_val = [
    "12",
    r"$10^{-3}$",
    "500",
    r"Constant $\times$ RBF",
    r"$10^{-6}$"
]

right_col = [
    r"AE damage latent dim",
    r"AE batch size",
    r"AE early stopping patience",
    r"GPR optimizer restarts",
    r"GPR normalise outputs"
]

right_val = [
    "12",
    "64",
    "30",
    "10",
    "True"
]

rows = [[l, lv, r, rv] for l, lv, r, rv in zip(left_col, left_val, right_col, right_val)]

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(10.5, 2.8), dpi=900)
ax.set_axis_off()

col_widths = [0.38, 0.24, 0.38, 0.12]

tbl = ax.table(
    cellText=rows,
    cellLoc="left",
    colWidths=col_widths,
    bbox=[0, 0.18, 1, 0.64]   # leave space for rules
)

# -----------------------------
# Styling
# -----------------------------
tbl.auto_set_font_size(False)
tbl.set_fontsize(15)
tbl.scale(1.0, 1.9)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor((0, 0, 0, 0))   # no gridlines
    cell.set_linewidth(0.0)
    cell.set_facecolor("white")
    cell.PAD = 0.02

# -----------------------------
# Blue top & bottom rules
# -----------------------------
rule_blue = (0/255, 114/255, 178/255)

fig.canvas.draw()
bbox = tbl.get_window_extent(fig.canvas.get_renderer())
bbox_ax = bbox.transformed(ax.transAxes.inverted())
x0, y0, x1, y1 = bbox_ax.x0, bbox_ax.y0, bbox_ax.x1, bbox_ax.y1

pad = 0.035 * (y1 - y0)

ax.plot([x0, x1], [y1 + pad, y1 + pad],
        color=rule_blue, linewidth=3.0,
        solid_capstyle="butt", transform=ax.transAxes)

ax.plot([x0, x1], [y0 - pad, y0 - pad],
        color=rule_blue, linewidth=3.0,
        solid_capstyle="butt", transform=ax.transAxes)

# -----------------------------
# Save
# -----------------------------
plt.savefig(
    "ae_gpr_hyperparameters.png",
    dpi=900,
    bbox_inches="tight",
    pad_inches=0.06
)
plt.close(fig)

print("Saved: ae_gpr_hyperparameters.png")
