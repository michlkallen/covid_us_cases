"""
Script to show how the COVID-19 cases in the US are changing over time.

Plot shows the new daily confirmed cases and is sorted by number of new cases.

To run: `python us_cases.py`
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.nonparametric.smoothers_lowess import lowess

# Style some of the plot parameters
mpl.rcParams['font.sans-serif'] = 'Clear Sans'
mpl.rcParams['lines.solid_capstyle'] = 'round'


def local_fit(y):
    """
    LOWESS fit of the data (set to 1 week fraction). Gives better view than rolling avg
    """
    x = np.arange(len(y))
    f = lowess(y, x, frac=1/7.)
    return f[:, 1]


def clean_plot(ax):
    """
    Cleans up the axes (removes spines, etc.)
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)


# Path to JHU data on Github
path = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/\
csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'

df_confirm = pd.read_csv(path)

# Tidy up the initial dataframe and add column for new cases
df = df_confirm.groupby('Province_State').sum()
df['delta'] = df.iloc[:, -1] - df.iloc[:, -2]

# Sort the new dataframe by new cases
df = df.sort_values(by='delta', ascending=False)

# ignore the cruise lines and some of the outlying territories (e.g., Guam)
exclude = ['American Samoa', 'Northern Mariana Islands', 'Diamond Princess',
           'Grand Princess', 'Guam', 'Virgin Islands']

df = df[~df.index.isin(exclude)].drop(columns='delta')

# List of states to include in the plot
states = list(df.index)
width, height = 5, len(states)*.5

fig, ax = plt.subplots(nrows=len(states), ncols=1,
                       figsize=(width, height),
                       tight_layout=True)

fig.subplots_adjust(hspace=0)

for i, state in enumerate(states):
    confirm = df.loc[state]
    # Calculate the daily change in number of cases
    daily = (confirm - confirm.shift(1))
    daily = daily[1:]

    ax[i].fill_between(daily.index, local_fit(daily.values), color='r', label=state)
    ax[i].annotate(f'{state} (yesterday: {int(daily.values[-1])})',
                   (0, .5), xycoords='axes fraction')
    ax[i].set_xlim(left=daily.index[0], right=daily.index[-1])
    ax[i].set_ylim(bottom=0)
    clean_plot(ax[i])

plt.savefig('us_cases.png', dpi=300, transparent=False)
