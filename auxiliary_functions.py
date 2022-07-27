import matplotlib.pyplot as plt
import numpy as np

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    
    # adapted from matplot lib example notbook
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    
    # adapted from matplot lib example notbook
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def correlation_ratio(df, categorical_feature, numerical_feature):
    """
    A function that computes the correlation ratio between a categorical and numerical features. 
    
    Parameters
    ----------
    df
        A dataframe with two columns comprising of the categorical and numerical features to be processed.
    categorical_feature
        The column name of the categorical feature.
    numerical_feature
        The columns name of the numerical feature.
    
    Returns
    -------
    correlation_ratio
        The correlation ratio between 0 and 1, where 0 means there are no differences between categories, 
        and 1 means that all of the differences can be attributed to the categorical differences.
    """
    
    if len(df.columns)<2:
        raise ValueError("There needs to be at least 2 features in the dataframe.")
        
    if categorical_feature not in df.columns:
        raise ValueError("Categorical feature no in dataframe.")
        
    if numerical_feature not in df.columns:
        raise ValueError("Numerical feature no in dataframe.")
     
    # retain only necessary features
    # accessing the df like this returns a copy, the original df is not altered
    df = df[[numerical_feature, categorical_feature]]
    
    # drop values where either feature is missing
    df.dropna(axis=0, how='any', inplace=True)
    
    # calculate correlation ratio
    categories = df[categorical_feature].unique()
    numerator = 0
    n_total = 0
    total_mean = df[numerical_feature].mean()
    
    for cat in categories:
        n_cat = (df[categorical_feature] == cat).sum()
        cat_mean = df[df[categorical_feature] == cat][numerical_feature].mean()
        
        numerator += n_cat * (cat_mean - total_mean)**2
        n_total += n_cat
    
    denominator = (n_total-1) * df[numerical_feature].var()
    
    return np.sqrt(numerator/denominator)
        
    
        