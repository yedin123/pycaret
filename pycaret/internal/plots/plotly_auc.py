# Author: Moez Ali
# AUC ROC curve using Plotly

def auc_plotly(estimator, X_train, y_train, X_test, y_test):
    """
    AUC plotly
    """

    import pandas as pd
    import numpy as np
    from yellowbrick.classifier import ROCAUC
    import plotly.express as px

    visualizer = ROCAUC(lr)
    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    
    l = []

    for i in visualizer.fpr.keys():
        x = visualizer.fpr[i]
        y = visualizer.tpr[i]
        df = pd.DataFrame()
        df['x'] = x 
        df['y'] = y
        df['Legend'] = 'ROC of ' + str(i) + ' AUC: ' + str(np.round(visualizer.roc_auc[i],4))
        l.append(df)

    data = pd.concat(l, axis=0)

    fig = px.line(x=data['x'], y=data['y'], template='plotly_dark', 
                color = data['Legend'],
                #width=800, height=400,
                labels={
                        "x": visualizer.ax.xaxis.axes.xaxis.get_label_text(),
                        "y": visualizer.ax.yaxis.axes.xaxis.get_label_text()
                    },)

    fig.update_layout(
                title={
                'text' : visualizer.ax.title.get_text(),
                'x':0.5,
                'xanchor': 'center',
                'font_size' : 20,
            })

    fig.show()