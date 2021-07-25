import numpy as np

def quitar_nulos( df ):
    for i in df.columns.tolist():
        if df[i].dtype!='object':
            if df[i].isnull().sum()>50000:
                df.dropna(subset=[i],inplace=True)
            elif df[i].isnull().sum()>0:
                df[i].replace(np.nan,df[i].mean(axis=0),inplace=True)
        else:
            df.dropna(subset=[i],inplace=True)
def remove_defectos(df, cols):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3-q1
        lower_limit = q1 - (IQR*1.5)
        upper_limit = q3 + (IQR*1.5)
        df.loc[df[col]<lower_limit,col] = lower_limit
        df.loc[df[col]>upper_limit,col] = upper_limit
def convert_obj_to_num(df, col):
    res = {}
    values = list(df[col].unique())
    for idx in range(len(values)):
        res[values[idx]] = idx
    return res