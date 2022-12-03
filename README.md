# Youtube Analyzer
─
Omkar Gokhale
PICT
Data Science, Web Development
omkargokhale2001@gmail.com

# Overview
The project is a website which allows youtubers to do a deep analysis of the last 20 videos posted on their channel. It contains three main sections:
The video projections section is used to generate interactive graphs to view the views, likes and  comments obtained on the last few videos.
The Best videos section is used to highlight which video performed the best based on views and user engagement.
The video optimizer gives the user tips for things like video duration as well as video name. It also has a video predictor which predicts the number of views a video could get using the video duration, video name, date and time of upload as parameters.

# Tech Stack
**Server:**<br />
The server and all the routes are written using Flask.<br />
**DataBase:**<br />
	We have used MongoDB as the Database.<br />
**Analysis:**<br />
	Python libraries like numpy, pandas are used for analyzing the data.<br />
**Machine Learning:**<br />
Sklearn are used  for model training, specifically the Support Vector Regressor, while nltk is used for natural language processing.<br />
**Data:**<br />
	The data is obtained in real time using the youtube API.<br />

# Flow of the Web App
When the user visits the website, they see a search bar where the youtuber name can be searched. Then the server checks if the user id is present in the database. If not, it sends a request to the youtube api using the search option. The channel id is obtained and then a request is sent to obtain data about the last 20 videos.
Then 3 options appear for further analysis:

**1. Video Projection:**<br />
	This section generates interactive graphs for the views, likes and number of comments on the videos using the plotly library.<br />
**2. Best Videos:**<br />
We use data analysis techniques to determine the best video based on view count and audience engagement using the like:view ratio.<br />
**3. Video Optimizer:**<br />
	Here we have converted all the string based data to numerical data. Then built a regressor for the numerical parameters. Then we have built a regressor for the video name using the tfidf vectorizer. We have used the Support Vector Regressor from sklearn for model training. Then we use the stacking regressor to obtain a combined model.
