#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('conda install -c conda-forge wikipedia --yes')


# In[19]:


import pandas as pd
import numpy as np
import wikipedia as wd


# In[20]:



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)


# ## Part1: Data Extraction from Wikipedia 

# In[21]:


html = wd.page("List of postal codes of Canada: M").html().encode("UTF-8")
df = pd.read_html(html, header = 0)[0]
df.head()


# Cells with Borough as "Not assigned" will be removed from the dataset </n>Cells with Neighborhood as NaN would be assigned with the same value as Borough

# In[22]:


df = df[df.Borough != 'Not assigned']
for index, row in df.iterrows():
    if row['Neighborhood'] == 'Not assigned':
        row['Neighborhood'] = row['Borough']
df = df.rename(columns={'Postal code': 'Postalcode'})
df.head()


# In[23]:


df = df.apply(lambda x: x.str.replace('/',',')).reset_index()
df.head()


# In[24]:


df 


# In[25]:


df.shape


# ## Part2: Adding Latitude and Longitude details 
# 

# In[26]:


# Using csv "http://cocl.us/Geospatial_data" to obtain latitude and longitude details of each location(Postal code)
Geo_data = pd.read_csv("http://cocl.us/Geospatial_data")
Geo_data.head()


# In[27]:


Geo_data= Geo_data.rename(columns={'Postal Code': 'Postalcode'})
toronto_df=pd.merge(df,Geo_data, on='Postalcode')
toronto_df.head()


# In[28]:


toronto_df


# ## Part3: Exploring neighborhoods in Toronto¶

# In[29]:


toronto_df[toronto_df['Postalcode'] == "M9V"]


# Importing necessary libraries for converting an address into geospatial address
# 

# In[15]:


get_ipython().system('conda install -c conda-forge geopy ')
from geopy.geocoders import Nominatim 
get_ipython().system('conda install -c conda-forge folium=0.5.0 ')
import folium


# In[ ]:





# In[30]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(toronto_df['Borough'].unique()),
        toronto_df.shape[0]
    )
)


# In[31]:



#Use geopy library to get the latitude and longitude values of toronto City.

address = 'Toronto'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
lat = location.latitude
lon = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(lat, lon))


# ### Map using folium indicating the neighborhoods of toronto

# In[34]:



map_toronto = folium.Map(location=[lat, lon], zoom_start=11)

# add markers to map
for lat, lng, borough, neighborhood in zip(toronto_df['Latitude'], toronto_df['Longitude'], toronto_df['Borough'], toronto_df['Neighborhood']):
    label = '{},{}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#76EE00',
        fill_opacity=0.6,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[35]:


CLIENT_ID = '4LCNKTZYUM2RG03U0UCYAZUAI2L4CDG12ACMJUWW3WWTXRHJ' # your Foursquare ID
CLIENT_SECRET = 'PYQ4OF40CTOY1JQAAEZGYQXLYFOSQ3QMZKYH10ODZKWFLFWI' # your Foursquare Secret
VERSION = '20200502' # Foursquare API version
radius = 1000
LIMIT = 200


# In[41]:


import io
import requests
radius=1000
url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}'.format(CLIENT_ID, CLIENT_SECRET, lat, lon, VERSION, radius)
results = requests.get(url).json()


# In[42]:



def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']

    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[43]:


import json
from pandas.io.json import json_normalize

venues = results['response']['groups'][0]['items']

nearby_venues = json_normalize(venues) # flatten JSON

filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[44]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[45]:



toronto_venues = getNearbyVenues(names=toronto_df['Neighborhood'],
                                   latitudes=toronto_df['Latitude'],
                                   longitudes=toronto_df['Longitude']
                                  )


# In[46]:


print(toronto_venues.shape)
toronto_venues.head()


# In[47]:


toronto_venues.groupby('Neighborhood').count()


# In[48]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# Obtained unique categories of venues in toronto 

# #### Analysing each neighborhood by exploring venue categories around the neighborhood
# 

# In[49]:


#Converting the categories into columns(one hot encoding)
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
# Adding neighborhood to the table
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood']

fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]
#Grouping neighborhood using frequency of occurance of each category
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[50]:


toronto_grouped.shape


# In[51]:


#print each neighboorhood with the top 5 most common venues
num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# Adding this data into a dataframe and get top 10 common venues for each neighborhood
# 

# In[52]:


# function to determine the top 10 common venues based on the frequency of occurance 
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[53]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# #### Clustering the neighborhoods

# In[54]:



from sklearn.cluster import KMeans
# set number of clusters
kclusters = 4

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[55]:



# add clustering labels

neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_df

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged


# In[82]:



import matplotlib.cm as cm
import matplotlib.colors as colors

map_clusters = folium.Map(location=[lat, lon], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels'].astype(int)):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
map_clusters


# EXAMINING each cluster 
# 

# In[80]:



# Cluster 1
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[83]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[84]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[85]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:




