import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.spatial.distance import hamming
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
# from surprise import SVD
# from surprise import Dataset
# from surprise.model_selection import cross_validate
import warnings; warnings.simplefilter('ignore')

   
# def load_css(file_name:str)->None:
#     """
#     Function to load and render a local stylesheet
#     """
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load_css("style_1.css")

# genres_list = ['None','Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science', 'Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV', 'Movie', 'Carousel', 'Productions', 'Vision', 'View', 'Entertainment', 'Telescene', 'Film', 'Group', 'Aniplex', 'GoHands', 'BROSTA', 'Mardock', 'Scramble', 'Production', 'Committee', 'Sentai', 'Filmworks', 'Odyssey', 'Media', 'Pulser', 'Rogue', 'State', 'The', 'Cartel']
# meta = pd.read_csv('https://raw.githubusercontent.com/saschakliegel/Imdb_project/main/meta.csv',error_bad_lines=False, header = 0)
# meta_cleaned = pd.read_csv('https://raw.githubusercontent.com/saschakliegel/Imdb_project/main/meta_cleaned.csv',error_bad_lines=False, header = 0)
# meta_sally = pd.read_csv('https://raw.githubusercontent.com/saschakliegel/Imdb_project/main/meta_sally.csv',header = 0)

def load_css(file_name:str)->None:           
     """
    Function to load and render a local stylesheet
     """
     with open(file_name) as f:
         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style_1.css")
                 

                    





genres_list = ['None','Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science', 'Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV', 'Movie', 'Carousel', 'Productions', 'Vision', 'View', 'Entertainment', 'Telescene', 'Film', 'Group', 'Aniplex', 'GoHands', 'BROSTA', 'Mardock', 'Scramble', 'Production', 'Committee', 'Sentai', 'Filmworks', 'Odyssey', 'Media', 'Pulser', 'Rogue', 'State', 'The', 'Cartel']

meta = pd.read_csv('https://raw.githubusercontent.com/saschakliegel/Imdb_project/main/meta.csv', header = 0)
meta_cleaned = pd.read_csv('https://raw.githubusercontent.com/saschakliegel/Imdb_project/main/meta_cleaned.csv',header = 0)
meta_sally = pd.read_csv('https://raw.githubusercontent.com/saschakliegel/Imdb_project/main/meta_sally.csv',header = 0)           

 



def main():

   
    selected_box = st.sidebar.selectbox(
    "Choose one of the sections provided",
    ('Exploratory Data Analysis','Content-based Filitering', 'Hybrid recommendation')

    )   
  
    if selected_box == 'Exploratory Data Analysis':
        Exploratory_Data_Analysis()
    if selected_box == 'Content-based Filitering':
        Content_based_Filitering()
    if selected_box == 'Hybrid recommendation':
        hybrid_recommendation()

    

    # select_genre = st.sidebar.selectbox("genres", options = lst)

def Exploratory_Data_Analysis():
        
    image = ("https://github.com/saschakliegel/Imdb_project/main/CalebsLogo.jpg")
    st.image(image, width=300)

    ### OPENING ---------------------------------------------------
    st.title("Visualization on TMDB Movie Dataset")
    st.markdown("**TMDb is really IMDb's competitor.** They are a community sourced movie and TV database. TMDb is not as old as IMDb and therefore does not have as much info. But unlike IMDb, TMDb has an open API allowing people freely access the information programmatically. Dataset has 45,466 Movies with 24 Features with few Nan values")
    st.markdown("Data containts 10866 Rows and 21 Columns, with minimal null value. The data cleaning process includes remove duplicate rows, changing format of release date into datetime format, remove unused columns which are irrelevant to the analysis process, remove movies which have zero value of budget and revenue")
    
    ### GRAPHS -- WORDCLOUD ----------------------------------------
    st.title("The Movie Dataset (TMDB) Visualization")
    st.subheader("Popular Words within Movie Titles")
    st.markdown("The word **Love, Girl, Day, Man** are also among the most commonly occuring words. I think this encapsulates the idea of the ubiquitious presence of romance in movies pretty well.")

    image = ("https://github.com/saschakliegel/Imdb_project/main/wordcloud_love.png")
    st.image(image, width=700)

    ### GRAPHS -- MOVIES OVER THE YEARS ---------------------------
    st.subheader("Number of Movies over the Years")
    st.markdown("The Dataset of 45,000 movies available to us does not represent the entire corpus of movies released since the inception of cinema. However, it is reasonable to assume that it does include almost every major film released in Hollywood as well as other major film industries across the world (such as Bollywood in India). With this assumption in mind, let us take a look at the number of movies produced by the year.")
    st.markdown("We notice that there is a **sharp rise in the number of movies starting the 1990s decade.** However, we will not look too much into this as it is entirely possible that recent movies were oversampled for the purposes of this dataset.")

    image = ("https://github.com/saschakliegel/Imdb_project/main/moviesovertheyears.png")
    st.image(image, width=700)

    ### GRAPHS -- GENRES ------------------------------------------
    st.subheader("Which Genre is the most popular, over the years ?")
    st.markdown("TMDB defines 32 different genres for our set of 45,000 movies")
    st.markdown("The proportion of movies of each genre has remained fairly constant since the beginning of this century except for Drama. **The proportion of drama films has fallen by over 5%. Thriller movies have enjoyed a slight increase in their share.**")

    image = ("https://github.com/saschakliegel/Imdb_project/main/genre_growth.png")
    st.image(image, width=700)

    ### GRAPHS -- LANGUAGE ----------------------------------------
    st.subheader("Which Languages are most popular, except English ?")
    st.markdown("There are over 93 languages represented in our dataset. **As we had expected, English language films form the overwhelmingly majority. French and Italian movies come at a very distant second and third respectively. Japanese and Hindi form the majority as far as Asian Languages are concerned.** Let us represent the most popular languages (apart from English) in the form of a bar plot.")

    image = ("https://github.com/saschakliegel/Imdb_project/main/language.png")
    st.image(image, width=700)

    ### GRAPHS -- Movies Released in a Particular Month --------------
    st.subheader("In which months do Hollywood Movies tend to release ?")
    st.markdown("**It appears that January is the most popular month when it comes to movie releases.** In Hollywood circles, this is also known as the the dump month when sub par movies are released by the dozen.")

    image = ("https://github.com/saschakliegel/Imdb_project/main/popularmonth.png")
    st.image(image, width=680)

    ### GRAPHS -- Blockbustor Movies Released in a Particular Month ------
    st.subheader("In which months do Blockbuster Movies tend to release ?")
    st.markdown("**It appears that the months of April, May and June** have the highest average gross among high grossing movies. This can be attributed to the fact that blockbuster movies are usually released in the summer when the kids are out of school and the parents are on vacation and therefore, the audience is more likely to spend their disposable income on entertainment.")

    image = ("https://github.com/saschakliegel/Imdb_project/main/popularmonth_blockbuster.png")
    st.image(image, width=700)

    ### GRAPHS -- Movies . Returns -----------------------------------------
    st.subheader("Do Some Months Tend to be More Successful than Others ?")
    st.markdown("**The months of June and July tend to yield the highest median returns. September is the least successful months on the aforementioned metrics.** Again, the success of June and July movies can be attributed to them being summer months and times of vacation. September usually denotes the beginning of the school/college semester and hence a slight reduction in the consumption of movies")

    image = ("https://github.com/saschakliegel/Imdb_project/main/popularmonth_yield.png")
    st.image(image, width=700) 

    ### GRAPHS -- POPULARTY vs VOTE AVERAGE -------------------------------
    st.subheader("Do Popularity and Vote Average share tangible relationship ?")
    st.markdown("Surprisingly, the Pearson Coefficient of the two aforementioned quantities is a measly 0.097 which suggests that **there is no tangible correlation. In other words, popularity and vote average and independent quantities.** It would be interesting to discover how TMDB assigns numerical popularity scores to its movies.")

    image = ("https://github.com/saschakliegel/Imdb_project/main/popularity_voteaverage.png")
    st.image(image, width=600)

    ### GRAPHS -- PRODUCTION COUNTRIES ------------------------------------
    st.subheader("Which countries serve as the most popular destinations for shooting movies by filmmakers ?")
    st.markdown("**The Full MovieLens Dataset consists of movies that are overwhelmingly in the English language (more than 31000)**. However, these movies may have shot in various locations around the world. It would be interesting to see which countries serve as the most popular destinations for shooting movies by filmmakers, especially those in the United States of America and the United Kingdom.")
    st.markdown("Unsurprisingly, the United States is the most popular destination of production for movies given that our dataset largely consists of English movies. Europe is also an extremely popular location with the UK, France, Germany and Italy in the top 5. Japan and India are the most popular Asian countries when it comes to movie production.")

    image = ("https://github.com/saschakliegel/Imdb_project/main/production_countries.png")
    st.image(image, width = 700)

    # ### GRAPHS -- DIRECTORS ------------------------------------------------
    # st.subheader("Directors who have raked in the most amount of money with their movies")
    # st.markdown("Which directors are the safest bet? For this, we will consider the average return brought in by a particular director. We will only consider those movies that have raked in at least 10 million dollars. Also, we will only consider actors and directors that have worked in at least 5 films.")


    ### CONCLUSION ---------------------------------------------------------
    st.markdown("**Thank you for exploring Movie Dataset with us thus far. We will use the insights to build recommendation system.**")

def Content_based_Filitering():
    
    def movie_recommender(distance_method, id, N):
        
        df_distance = pd.DataFrame(data=meta['id'])
        md_ver5 = meta.drop_duplicates(subset="id", keep="first")
        md_ver5 = md_ver5.set_index('id')
        df_distance = df_distance[df_distance['id'] != int(id)]
        try:
            df_distance['distance'] = df_distance['id'].apply(lambda x: distance_method(np.array(md_ver5.loc[x]),np.array(md_ver5.loc[int(id)])))
        except:
            df_distance['distance'] = df_distance['id'].apply(lambda x: distance_method(np.array(md_ver5.loc[x].iloc[0]),np.array(md_ver5.loc[int(id)])))
        df_distance.sort_values(by='distance', inplace=True)
    
        rec = df_distance.head(N)  
    
        l1 = rec['id'].to_list()
        recom = ()
        for i in l1:
            # index = int(index)
            idx = meta_cleaned[meta_cleaned['id']==str(i)]
            # print(idx)
            title = idx["original_title"].to_string(index=False)
            date = idx["release_date"].to_string(index=False)
            date = date[-4:]
            genre = idx["genres"].to_string(index=False)
            genre_toprint = genre.replace("[","").replace("]","").replace("'","")
            recom = "Title: ", title ,  " Release Date: ", date, "\n" ,"Genre: ", genre_toprint
            
            print(recom)
        return l1                     
    
    # movie_recommender(hamming,choice_id,10)

    st.title(":popcorn:Movie Recommender:popcorn:")   
    meta_cleaned = pd.read_csv('https://raw.githubusercontent.com/saschakliegel/Imdb_project/main/meta_cleaned.csv', header = 0)
    list_of_choices = list(meta_cleaned["original_title"])

    list_of_choices.append(" ")
    movie_choice = st.selectbox("Choose a movie",list_of_choices, index=len(list_of_choices)-1)
    
    genre = st.sidebar.selectbox('Select the genre', (genres_list), index=0)
    year  = st.sidebar.slider('Select a range of values',1900, 2021, (1900,1900))
    min_year , max_year = year
    

    if movie_choice != " ":  
        choice_id = meta_cleaned[meta_cleaned['original_title'] == movie_choice]['id']
        result_list = movie_recommender(hamming,choice_id,10)
        if min_year != 1900 & max_year != 1900:
            meta_cleaned = meta_cleaned[meta_cleaned[(meta_cleaned["release_date"].to_string(index=False)[-4:] >= str(min_year)) & (meta_cleaned["release_date"].to_string(index=False)[-4:] <= str(max_year))]]
            st.write('Range:', year)
        if genre == "None" :
            
            for i in result_list:
                movie_title = meta_cleaned[meta_cleaned['id'] == str(i)]['original_title'].to_string(index=False)
                st.markdown(movie_title)
        else:
            for i in result_list:
                if int(meta[meta['id'] == int(meta_cleaned[meta_cleaned['id'] == str(i)]['id'])][genre] ==1):
                    movie_title = meta_cleaned[meta_cleaned['id'] == str(i)]['original_title'].to_string(index=False)
                    st.markdown(movie_title)
                
        
            
            
        st.markdown(movie_title)
        # st.markdown("""<style>.big-font {font-size:300px !important;}</style>""", unsafe_allow_html=True)
        # st.markdown('<p calss="big-font"> movie_title </p>', unsafe_allow_html=True )   
        
#st.text(movie_recommender(hamming,choice_id,10)) 
# 
# Sally's code
# meta = pd.read_csv('meta.csv',index_col=[0])
rate = pd.read_csv('https://raw.githubusercontent.com/saschakliegel/Imdb_project/main/ratings_small.csv')

def hybrid_recommendation():
    def contentbased(title):
        #content-based
        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(meta_sally['soup'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        #meta = meta.reset_index()
        ids = meta_sally['id']
        indices = pd.Series(meta_sally.index, index=meta_sally['id'])
        movieid = int(meta_sally[meta_sally['title']==title]['id'])
        idx = indices[movieid]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1],reverse=True)
        sim_scores = sim_scores[1:101]
        movie_indices = [i[0] for i in sim_scores]
        content_based = ids.iloc[movie_indices].to_list()

        return content_based

    def hybrid(user,title):
        content_based = contentbased(title)
        ratings_filtered = rate[rate["movieId"].isin(content_based)]
        ratings_filtered = ratings_filtered.append(rate[rate['userId']==user])
        ratings = ratings_filtered

        userItemRatingMatrix=pd.pivot_table(ratings, values='rating',
                                    index=['userId'], columns=['movieId'])

        allUsers = pd.DataFrame(userItemRatingMatrix.index)
        allUsers = allUsers[allUsers.userId!=user]
        # Add a column to this df which contains distance of active user to each user
        allUsers["distance"] = allUsers["userId"].apply(lambda x: hamming(userItemRatingMatrix.loc[user],userItemRatingMatrix.loc[x]))
        KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["userId"][:10]
        # get the ratings given by nearest neighbours
        NNRatings = userItemRatingMatrix[userItemRatingMatrix.index.isin(KnearestUsers)]
        # Find the average rating of each book rated by nearest neighbours
        avgRating = NNRatings.apply(np.nanmean).dropna()
        # drop the books already read by active user
        #moviesAlreadywatched = userItemRatingMatrix.loc[user].dropna().index
        #avgRating = avgRating[~avgRating.index.isin(moviesAlreadywatched)]
        topNmovieIds = avgRating.sort_values(ascending=False).index[:11].to_list()

        return topNmovieIds
           
    st.title("Hybrid Recommender")

    movie_list = list(meta_sally["title"])
    movie_list.append(" ")

    user_list = []
    for user in rate['userId']:
        if user not in user_list:
            user_list.append(user)
    user_list.append(" ")       

    user_choice = st.selectbox("Choose a user",user_list, index=len(user_list)-1)
    movie_choice = st.selectbox("Choose a movie",movie_list, index=len(movie_list)-1)

    if movie_choice != " " and user_list != " ":
        movie_title = meta_sally[meta_sally['title'] == movie_choice]['title'].to_string(index=False)
        result_list = hybrid(user_choice,movie_title)

        for i in result_list:
            movie_title = meta_sally[meta_sally['id'] == i]['title'].to_string(index=False)
            
            st.text(movie_title)          

if __name__ == "__main__":
    main()








# def movie_recommender(distance_method, id, N):
    
#     df_distance = pd.DataFrame(data=meta['id'])
#     md_ver5 = meta.drop_duplicates(subset="id", keep="first")   
#     md_ver5 = md_ver5.set_index('id')
#     df_distance = df_distance[df_distance['id'] != int(id)]
#     try:
#         df_distance['distance'] = df_distance['id'].apply(lambda x: distance_method(np.array(md_ver5.loc[x]),np.array(md_ver5.loc[int(id)])))
#     except:
#         df_distance['distance'] = df_distance['id'].apply(lambda x: distance_method(np.array(md_ver5.loc[x].iloc[0]),np.array(md_ver5.loc[int(id)])))
#     df_distance.sort_values(by='distance', inplace=True)
   
#     rec = df_distance.head(N)

#     l1 = rec['id'].to_list()
#     recom = ()
#     for i in l1:
#         # index = int(index)
#         idx = meta_cleaned[meta_cleaned['id']==str(i)]
#         # print(idx)
#         title = idx["original_title"].to_string(index=False)
#         date = idx["release_date"].to_string(index=False)
#         genre = idx["genres"].to_string(index=False)
#         genre_toprint = genre.replace("[","").replace("]","").replace("'","")
#         recom = "Title: ", title ,  " Release Date: ", date, "\n" ,"Genre: ", genre_toprint
#         print(recom)
#     return l1

# # movie_recommender(hamming,choice_id,10)

# st.title(":popcorn: *Movie Recommender*  :popcorn:")
# st.header(":sunglasses: Choose A Movie  :sunglasses:")

# list_of_choices = list(meta_cleaned["original_title"])

# list_of_choices.append(" ")
# movie_choice = st.selectbox(" ",list_of_choices, index=len(list_of_choices)-1)

# if movie_choice != " ":
#     choice_id = meta_cleaned[meta_cleaned['original_title'] == movie_choice]['id']
#     result_list = movie_recommender(hamming,choice_id,10)

#     for i in result_list:
#         movie_title = meta_cleaned[meta_cleaned['id'] == str(i)]['original_title'].to_string(index=False)
        
#         st.text(movie_title)

# #st.text(movie_recommender(hamming,choice_id,10))
