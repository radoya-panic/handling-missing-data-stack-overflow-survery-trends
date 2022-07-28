import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss

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
        The column name of the numerical feature.
    
    Returns
    -------
    The correlation ratio between 0 and 1, where 0 means there are no differences between categories, 
    and 1 means that all of the differences can be attributed to the categorical differences.
    """
    
    if len(df.columns)<2:
        raise ValueError("There needs to be at least 2 features in the dataframe.")
        
    if categorical_feature not in df.columns:
        raise ValueError("Categorical feature not in dataframe.")
        
    if numerical_feature not in df.columns:
        raise ValueError("Numerical feature not in dataframe.")
     
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

def uncertainty_coefficient(df, x, y):
    """
    The function uncertainty_coefficient(df, x, y) returns the uncertainty coefficient U(X|Y). This should tell you
    the fraction of X we can predict given Y. This is also known as the entropy coefficient.
    
    Parameters
    ----------
    df
        A dataframe with two columns comprising of the categorical features to be processed.
    x
        The column name of one of the categorical feature.
    y
        The column name of the other categorical feature.  
    
    Returns
    -------
    The uncertainty coefficient which is between 0 and 1. 
    """
    if len(df.columns)<2:
        raise ValueError("There needs to be at least 2 features in the dataframe.")
        
    if x not in df.columns:
        raise ValueError("The categorical feature {0} is not in the dataframe.".format(x))
        
    if y not in df.columns:
        raise ValueError("The categorical feature {0} is not in the dataframe.".format(y))
    
    # attain a copy of just the two features you need
    df = df[[x,y]]
    
    # drop values where either feature is missing
    df.dropna(axis=0, how='any', inplace=True)
    
    H_X = entropy(df,x)
    H_X_Y = conditional_entropy(df,x,y)
    
    return (H_X - H_X_Y)/H_X

def entropy(df,x):
    """
    The function entropy(df,x) returns the entropy E(X). It is the average surprise/information conveyed per event. 
    
    Parameters
    ----------
    df
        A dataframe with two columns comprising of the categorical features to be processed.
    x
        The column name of one of the categorical feature.
    
    Returns
    -------
    The entropy. 
    """
    
    if len(df.columns)<2:
        raise ValueError("There needs to be at least 2 features in the dataframe.")
        
    if x not in df.columns:
        raise ValueError("The categorical feature {0} is not in the dataframe.".format(x))
    
    # attain a copy of just the feature you need
    df = df[x]
    
    # drop values where the feature is missing
    df.dropna(axis=0, how='any', inplace=True)

    # calculate the probability distribution 
    probs = np.array(df.value_counts())/len(df)
    
    # return the entropy, which is the expected value of -log(P(X))
    return ss.entropy(probs)

def conditional_entropy(df,x,y):
    """
    The function conditional_entropy(df,x) returns the entropy E(X|Y). It is the average surprise/information 
    conveyed per event provided Y. 

    Parameters
    ----------
    df
        A dataframe with two columns comprising of the categorical features to be processed.
    x
        The column name of one of the categorical feature.
    y
        The column name of the other categorical feature.  

    Returns
    -------
    The conditional entropy. 
    """
    if len(df.columns)<2:
        raise ValueError("There needs to be at least 2 features in the dataframe.")

    if x not in df.columns:
        raise ValueError("The categorical feature {0} is not in the dataframe.".format(x))

    if y not in df.columns:
        raise ValueError("The categorical feature {0} is not in the dataframe.".format(y))

    # attain a copy of just the two features you need
    df = df[[x,y]]

    # drop values where either feature is missing
    df.dropna(axis=0, how='any', inplace=True)

    # create a contingency table and calculate the joint probability dsitribution
    p_xy = pd.crosstab(df[x],df[y])
    p_xy = p_xy/len(df)
    
    # calculate the probability distribution of just y
    p_y = df[y].value_counts()/len(df)
    
    # calculate the conditionaly entroy 
    H_X_Y = 0
    for i in p_xy.index:
        for j in p_xy.columns:
            if p_xy.loc[i,j] != 0:
                H_X_Y += -1 * p_xy.loc[i,j] * np.log(p_xy.loc[i,j]/p_y.loc[j])
    
    return H_X_Y
