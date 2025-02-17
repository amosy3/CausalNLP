def predict_concepts(df, concepts):
    for c in concepts:
        #Train a model to predict c usit Train split
        #log model

        df['%s_predicted' % c] = model(X) #apply on Causal estimation split
        #save updated dataset
