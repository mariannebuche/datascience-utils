import pandas as pd


def description_var_quali(df, id_var, population_var, description_var):
    """
    Output categorical feature distribution by population.

    Parameters
    ----------
    df : pandas.DataFrame
        input DataFrame
    id_var : str
        id feature
    population_var : str
        population feature
    description_var : str
        description feature

    Returns
    -------
    pandas.DataFrame

    """

    data = df.copy()

    result = data.groupby([population_var, description_var])[id_var] \
        .count().reset_index().pivot_table(
        index=description_var, columns=population_var, values=id_var) \
        .reset_index().rename({description_var: 'value'}, axis=1)

    result['variable'] = description_var
    result = result[['variable', 'value'] +
                    list(sorted(data[population_var].unique()))]
    for i in list(sorted(data[population_var].unique())):
        result[i] = result[i].fillna(0).astype(int) \
                    / len(data[data[population_var] == i])

    return result


def description_var_quanti(df, id_var, population_var, description_var,
                           slices=4, precision=0):
    """
    Output numerical feature description by event population.

    More precisely, get basic distribution statistics of the numerical feature
    then split it into slices and output distribution of the split feature
    by event population.

    Parameters
    ----------
    df : pandas.DataFrame
        input DataFrame
    id_var : str
        id feature
    population_var : str
        population feature
    description_var : str
        description feature
    slices : int
        (optional, default 4) in how many slices to cut the variable
    precision : int
        (optional, default 0) precision of the slices bins

    Returns
    -------
    pandas.DataFrame

    """

    data = df.copy()
    data[description_var + '_class'] = pd.qcut(data[description_var],
                                               q=slices,
                                               precision=precision,
                                               duplicates='drop')
    result = description_var_quali(data, id_var, population_var,
                                   description_var + '_class')

    missing = pd.DataFrame(data.groupby(population_var)[description_var].apply(
        lambda x: x.isnull().values.sum())).reset_index() \
        .rename(columns={description_var: 'missing'})
    description = data.groupby(population_var)[description_var].describe() \
        .reset_index().join(missing.set_index(population_var),
                            on=population_var, how='outer')
    description = description[[population_var, 'missing', 'count'] +
                              [col for col in description.columns
                               if col not in ([population_var,
                                               'count', 'missing'])]]
    description_cols = [col for col in description.columns
                        if col != population_var]
    for col in description_cols:
        temp_result = pd.pivot_table(
            description, values=col, columns=population_var) \
            .reset_index().rename(columns={'index': 'value'})
        temp_result['variable'] = description_var
        temp_result = temp_result[['variable', 'value'] + list(
            sorted(data[population_var].unique()))]
        result = pd.concat([result, temp_result], axis=0, join='outer',
                           ignore_index=True, sort=False)

    return result


def describe_dataframe(df, features, id_var, population_var, max_num_values=10):
    """
    Describe selected dataframe variables by population.

    Parameters
    ----------
    df : pandas.DataFrame
        input DataFrame
    features : list of str
        list of features to analyze
    id_var : str
        id feature
    population_var : str
        event feature
    max_num_values : int
        max nbr of unique values to analyze a numerical feature as categorical

    Returns
    -------
    pandas.DataFrame

    """

    data = df.copy()
    result = pd.DataFrame()

    for i in features:
        if (data[i].dtypes == 'object') | (len(data[i].unique()) <= max_num_values):
            data[i] = data[i].fillna('(NA)')
            result = result.append(
                description_var_quali(data, id_var, population_var, i),
                ignore_index=True, sort=False)

        if (data[i].dtypes != 'object') & (
                len(data[i].unique()) > max_num_values):
            result = result.append(
                description_var_quanti(data, id_var, population_var, i),
                ignore_index=True, sort=False)

    count = pd.DataFrame(data.groupby(population_var)[id_var].count()).pivot_table(
        columns=population_var, values=id_var).reset_index().drop('index', axis=1)
    volume = pd.concat(
        [pd.DataFrame({'variable': 'total', 'value': 'volume'}, index=[0]),
         count], axis=1, sort=False)
    volume = volume[
        ['variable', 'value'] + list(sorted(data[population_var].unique()))]

    result = pd.concat([volume, result], axis=0, join='outer',
                       ignore_index=True, sort=False)
    result = result[
        ['variable', 'value'] + list(sorted(data[population_var].unique()))]

    return result


def export_profiling(df, export_path, file_name):
    """

    Export profiling DataFrame to a semicolon separated csv file.

    Parameters
    ----------
    df : pandas.DataFrame
        input DataFrame with profiling info
    export_path : str
        location for file export (example: 'C:/my_folder/)
    file_name : str
        name of the exported file (example: 'exported_profiling')

    Returns
    -------
    None

    """

    df.to_csv(export_path + file_name + '.csv', sep=';', header=True,
              index=False, encoding='utf-8-sig', line_terminator='\n')
