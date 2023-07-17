from flask import Flask, render_template, request, redirect, session
import pandas as pd
import numpy as np
import pipeline
import datetime




app = Flask(__name__)
''' 
@app.route("/")
def hello():
    return "Le serveur flask fonctionne"
'''

#Affichage de IHM via recommendation.html
@app.route('/')
def view_form():
    
    df_clicks=pd.read_csv("input/clicks2.csv")
    nbusers=df_clicks["user_id"].nunique()
    nbarticles=df_clicks["click_article_id"].nunique()
    #max_article_nb=max(df_clicks["click_article_id"])+1
    #max_userid_nb=max(df_clicks["user_id"])+1
    
    return render_template('recommendations.html',nbusers=nbusers,nbarticles=nbarticles)#, max_article_nb=max_article_nb, max_userid_nb= max_userid_nb)
 

#utilisation de la méthode POST
@app.route('/predict_post', methods=['POST'])
def predict_post():
    if request.method == 'POST':
        
        df_clicks=pd.read_csv("input/clicks2.csv")
        nbusers=df_clicks["user_id"].nunique()
        nbarticles=df_clicks["click_article_id"].nunique()
        #max_article_nb=max(df_clicks["click_article_id"])+1
        #max_userid_nb=max(df_clicks["user_id"])+1
        
        nbusers=df_clicks["user_id"].nunique()
        nbarticles=df_clicks["click_article_id"].nunique()
        userID = request.form['message']
        userID=int(userID)

        csr_item_user, csr_user_item = pipeline.compute_interaction_matrix(df_clicks)
        
        #pipeline.get_cf_reco(df_clicks, userID, csr_item_user, csr_user_item,model_path=None, n_reco=5, train=True)
        recommendations=pipeline.get_cf_reco(df_clicks, userID, csr_item_user, csr_user_item, model_path="./recommender.model", n_reco=5, train=False) #
        
        #recoms=list(recommendations[0])
    
    return render_template('recommendations.html', recoms=recommendations, userID=userID,nbusers=nbusers)#,nbarticles=nbarticles, max_article_nb=max_article_nb, max_userid_nb= max_userid_nb)
   
#utilisation de la méthode POST
@app.route('/ajout_article', methods=['POST'])
def ajout_article():
    
    if request.method == 'POST':
        
        # chargement du fichier pickle contenant l'embedding des articles sans les word count =0 et sans les articles sans lecteurs (0 interactions)
        df_embeddings=pd.read_csv("input/df_embeddings_inter.csv", index_col=0)
        
        #Chargement du fichier des interactions
        df_clicks=pd.read_csv("input/clicks2.csv")
        nbusers=df_clicks["user_id"].nunique()
        nbarticles=df_clicks["click_article_id"].nunique()
        #max_article_nb=max(df_clicks["click_article_id"])+1
        #max_userid_nb=max(df_clicks["user_id"])+1
        
        newarticle= request.form['message']
        newarticle=int(newarticle)
        
        #Mise à jour de la dataframe en incluant 30000 click sur le nouvel article ajouté
        #df=df_clicks.sample(n = 30000)
        
        '''Etant donné que l'on a pas la formule qui a permis de créer le ficher embedding, on va créer alétoirement un ligne embedding
        #puis trouver son plus proche voisin que l'on va appeller y'''
        
    
        #Epurage de df_embedding pour ne garder que les articles ayant eu au moins 1 interaction
        #Création de la liste des articles ayant eu au moins 1 interaction
        list_article_inter=[]
        list_article_inter=df_clicks["click_article_id"].unique()
        print(len(list_article_inter))

        #Création de l'embedding ne contenant que les articles ayant eu au moins 1 interaction
        df_embeddings_inter= df_embeddings[df_embeddings.index.isin(list_article_inter)]
        print(len(df_embeddings_inter))
        
        #Recherche du plus proche voisin de l'embedding du nouvel article crée aléatoirement
        y=pipeline.proche_voisin(newarticle,df_embeddings_inter)
        
        #Récupération de toutes les lignes d'interactions du meilleur voisin que je vais pouvoir cloner avec le nom du nouvelle article
        df=df_clicks.loc[df_clicks["click_article_id"]==y]
        print("On va jouter",len(df),"interactions")
        #display(df.info())
        #display(df.head())
        
        #Affectation du nom du nouvel article et mise à jour de la date
        df["click_article_id"]=newarticle
        df["click_timestamp"]=datetime.datetime.today()
        df_clicks2 = pd.concat([df_clicks, df], axis=0)#,ignore_index=True)
        df_clicks2.to_csv("input/clicks2.csv",index=False)
        df_clicks=df_clicks2.copy()
        userID=55555 #Au hasard pour respecter la structure de la fonction (on aurait pu prendre un autre user)
        
        #Recalcul de la matrice d'interaction et réentrainement du modèle
        csr_item_user, csr_user_item = pipeline.compute_interaction_matrix(df_clicks)
        pipeline.get_cf_reco(df_clicks, userID, csr_item_user, csr_user_item,model_path=None, n_reco=5, train=True)
        
        #Recalcul des paramètres pour affichage dans html
        df_clicks=pd.read_csv("input/clicks2.csv")
        nbarticles=df_clicks["click_article_id"].nunique()
        

    return render_template('recommendations.html',newarticle=newarticle,nbusers=nbusers,nbarticles=nbarticles)#, max_article_nb=max_article_nb, max_userid_nb= max_userid_nb)
 
    
    
#utilisation de la méthode POST
@app.route('/ajout_utilisateur', methods=['POST'])
def ajout_utilisateur():
    
    if request.method == 'POST':
        
        df_clicks=pd.read_csv("input/clicks2.csv")
        nbusers=df_clicks["user_id"].nunique()
        nbarticles=df_clicks["click_article_id"].nunique()
        #max_article_nb=max(df_clicks["click_article_id"])+1
        #max_userid_nb=max(df_clicks["user_id"])+1
        
        newuser= request.form['message']
        newuser=int(newuser)
        
        #Mise à jour de la dataframe en incluant 50 interactions sur le nouvel utilisateur ajouté
        df=df_clicks.sample(n = 50) #ajout de 50 interactions par utilisateur
        df["user_id"]= newuser
        df["click_timestamp"]=datetime.datetime.today()
        df_clicks2 = pd.concat([df_clicks, df], axis=0)#,ignore_index=True)
        df_clicks2.to_csv("input/clicks2.csv",index=False)
        df_clicks=df_clicks2.copy()
        userID=newuser
        
        #Recalcul de la matrice d'interaction et réentrainement du modèle
        csr_item_user, csr_user_item = pipeline.compute_interaction_matrix(df_clicks)
        pipeline.get_cf_reco(df_clicks, userID, csr_item_user, csr_user_item,model_path=None, n_reco=5, train=True)
        
        
        #Recalcul des paramètres pour affichage dans html
        df_clicks=pd.read_csv("input/clicks2.csv")
        nbusers=df_clicks["user_id"].nunique()
       

    return render_template('recommendations.html',newuser=newuser,nbusers=nbusers,nbarticles=nbarticles)#, max_article_nb=max_article_nb, max_userid_nb= max_userid_nb)    
    
if __name__ == '__main__':
    app.run(port=7318, debug=True)

    