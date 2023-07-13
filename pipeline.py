from scipy.sparse import csr_matrix
import pandas as pd
import pickle
from implicit.bpr import BayesianPersonalizedRanking
from flask import jsonify
import json







def compute_interaction_matrix(clicks):
    # Création de la dataframe d'interaction entre users et articles
    interactions = clicks.groupby(['user_id','click_article_id']).size().reset_index(name='count')
    print('Interactions DF shape: ', interactions.shape)

    # csr = compressed sparse row (format adapté au opérations mathématiques sur les lignes )
    # Création de la sparse matrix de taille (number_items, number_user)
    csr_item_user = csr_matrix((interactions['count'].astype(float),
                                (interactions['click_article_id'],
                                 interactions['user_id'])))
    print('CSR Shape (number_items, number_user): ', csr_item_user.shape)
    
    # Création de la sparse matrix de taille (number_user, number_items)
    csr_user_item = csr_matrix((interactions['count'].astype(float),
                                (interactions['user_id'],
                                 interactions['click_article_id'])))
    print('CSR Shape (number_user, number_items): ', csr_user_item.shape)
    
    
    return csr_item_user, csr_user_item



def get_cf_reco(clicks, userID, csr_item_user, csr_user_item, model_path=None, n_reco=5, train=True):#

    #start = time()
    # Entrainement du modele sur la sparse matrix de taille (number_items, number_user)
    
    if train or model_path is None:
        #model = LogisticMatrixFactorization(factors= 128, random_state=42)
        model = BayesianPersonalizedRanking(factors=100, regularization=0.01, use_gpu=False, iterations=5, random_state=42)
        print("[INFO] : Début de l'entrainement du modèle")
        model.fit(csr_user_item)

        # Enregistrement du modèle sur le PC
        with open('recommender.model', 'wb') as filehandle:
            pickle.dump(model, filehandle)
    else:
        with open('recommender.model', 'rb') as filehandle:
            model = pickle.load(filehandle)
            print(model)
            
  
    # Recommandation de N articles depuis la sparse matrix de taille (number_user, number_items)
    # Utilisation de Implicit built-in method
    # N (int) : nombre d'article à recommander
    # filter_already_liked_items (bool) : Si true, ne pas retourner d'articles présent...   
    # ...dans le traing set qui ont déjà été consulté par le user
    recommendations_list = []
    
    recommendations = model.recommend(userID, csr_user_item[userID], N=n_reco, filter_already_liked_items=True)

    #print(f"[INFO] : Temps d'entrainement {round(time() - start, 2)}s")
    
    #recommendations = [elt[0] for elt in recommendations]
    recommendations = [elt[:n_reco] for elt in recommendations]
    
    recoms=recommendations[0].tolist()
    
    #return recommendations
    #return jsonify(recoms)
    return  json.dumps(recoms)




    
    