

def plot_hist(feature_values, feature_name='', fill_color='', line_color=''):
    import numpy as np
    from bokeh.io import show, output_notebook, show
    from bokeh.plotting import figure
    from bokeh.colors import named
    from random import randint

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
