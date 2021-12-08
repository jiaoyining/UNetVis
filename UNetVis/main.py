import panel as pn
import panel.widgets as pnw
from matplotlib.backends.backend_agg import FigureCanvas
import hvplot.pandas

def hvplot(avg, highlight):
    return avg.hvplot(height=200) * highlight.hvplot.scatter(color='orange', padding=0.1)

import ipywidgets as ipw
pn.extension()

pn.extension('ipywidgets')
date   = ipw.DatePicker(description='Date')
slider = ipw.FloatSlider(description='Float')
play   = ipw.Play()

layout = ipw.HBox(children=[date, slider, play])

pn.panel(layout).servable()
