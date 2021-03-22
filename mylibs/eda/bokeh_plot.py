import numpy as np
from bokeh.io import show, output_notebook, show
from bokeh.plotting import figure
from bokeh.colors import named
from random import randint

def plot_hist(feature_values, feature_name='', fill_color='', line_color=''):
    #print(random.randint(0,len(dir(named)[10:])))
    color_list = dir(named)[10:]

    # output to notebook
    output_notebook()

    hist, edges = np.histogram(feature_values, density=True, bins=50)

    x = np.linspace(-2, 2, 1000)

    p = figure(plot_height = 600, plot_width = 600,
           title = 'Histogram of {}'.format(feature_name),
          x_axis_label = feature_name,
           y_axis_label = 'count')

    if not line_color:
        # import random
        # line_color = random.choice(color_list)
        line_color = color_list[randint(0,len(color_list))-1]
    if not fill_color:
        fill_color = color_list[randint(0,len(color_list))-1]

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=fill_color, line_color=line_color)

    #output_file("hist.html")
    show(p)

    
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs


class BokehHistogram():

    def __init__(self, colors=["SteelBlue", "Tan"], height=400, width=600):
        self.colors = colors
        self.height = height
        self.width = width
        output_notebook()

    def hist_hover(self, dataframe, column, bins=30, log_scale=False, show_plot=True):
        hist, edges = np.histogram(dataframe[column], bins = bins)
        hist_df = pd.DataFrame({column: hist,
                                 "left": edges[:-1],
                                 "right": edges[1:]})
        hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                               right in zip(hist_df["left"], hist_df["right"])]

        if log_scale == True:
            hist_df["log"] = np.log(hist_df[column])
            src = ColumnDataSource(hist_df)
            plot = figure(plot_height = self.height, plot_width = self.width,
                  title = "Histogram of {}".format(column.capitalize()),
                  x_axis_label = column.capitalize(),
                  y_axis_label = "Log Count")    
            plot.quad(bottom = 0, top = "log",left = "left", 
                right = "right", source = src, fill_color = self.colors[0], 
                line_color = "black", fill_alpha = 0.7,
                hover_fill_alpha = 1.0, hover_fill_color = self.colors[1])
        else:
            src = ColumnDataSource(hist_df)
            plot = figure(plot_height = self.height, plot_width = self.width,
                  title = "Histogram of {}".format(column.capitalize()),
                  x_axis_label = column.capitalize(),
                  y_axis_label = "Count")    
            plot.quad(bottom = 0, top = column,left = "left", 
                right = "right", source = src, fill_color = self.colors[0], 
                line_color = "black", fill_alpha = 0.7,
                hover_fill_alpha = 1.0, hover_fill_color = self.colors[1])

        hover = HoverTool(tooltips = [('Interval', '@interval'),
                                  ('Count', str("@" + column))])
        plot.add_tools(hover)

        if show_plot == True:
            show(plot)
        else:
            return plot

    def histotabs(self, dataframe, features, log_scale=False, show_plot=False):
        hists = []
        for f in features:
            h = self.hist_hover(dataframe, f, log_scale=log_scale, show_plot=show_plot)
            p = Panel(child=h, title=f.capitalize())
            hists.append(p)
        t = Tabs(tabs=hists)
        show(t)

    def filtered_histotabs(self, dataframe, feature, filter_feature, log_scale=False, show_plot=False):
        hists = []
        for col in dataframe[filter_feature].unique():
            sub_df = dataframe[dataframe[filter_feature] == col]
            histo = self.hist_hover(sub_df, feature, log_scale=log_scale, show_plot=show_plot)
            p = Panel(child = histo, title=col)
            hists.append(p)
        t = Tabs(tabs=hists)
        show(t)
        
#https://towardsdatascience.com/interactive-histograms-with-bokeh-202b522265f3
