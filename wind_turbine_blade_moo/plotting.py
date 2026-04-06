import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
TAB10_COLORS = tuple(plt.get_cmap('tab10').colors)
VIRIDIS_CMAP = ListedColormap(plt.get_cmap('viridis')(range(256)))

def apply_publication_style():
    rcParams.update({'figure.figsize': (10.0, 6.0), 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08, 'font.size': 13, 'axes.titlesize': 17, 'axes.labelsize': 14, 'axes.titleweight': 'semibold', 'axes.labelweight': 'medium', 'axes.labelpad': 10.0, 'axes.linewidth': 1.1, 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'axes.edgecolor': '#1A1A1A', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#D7D7D7', 'grid.linewidth': 0.8, 'grid.alpha': 0.55, 'grid.linestyle': '-', 'lines.linewidth': 2.8, 'lines.markersize': 7.0, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'xtick.major.pad': 7.0, 'ytick.major.pad': 7.0, 'xtick.major.size': 5.0, 'ytick.major.size': 5.0, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'legend.fontsize': 11, 'legend.frameon': True, 'legend.facecolor': 'white', 'legend.edgecolor': '#BDBDBD', 'axes.prop_cycle': plt.cycler(color=TAB10_COLORS)})

def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which='major')
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which='minor', linewidth=0.45, alpha=0.22)
    ax.tick_params(axis='both', which='major', pad=7)
