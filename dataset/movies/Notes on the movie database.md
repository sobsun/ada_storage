## Notes on the movie database

# plot_summaries.txt
WikipediaID       plot_summaries
plots summaries are quite long and the real info on those => we need to somehow include it in the course analysis
approx 176 words

# corenlp_plot_summaries.tar.gz
multiple files ? I don't have this one

# movie.metadata.tsv
the genre/country/languages are always available => so could be a good source
genre = 365 non-exclusive categories
80 000 movies
à vue de nez looks relatively nicely distributed 1950-2010 (entre 250 et 1000 films par année)
même si forcément y en a plus plus on avance dans le temps

# character.metadata.tsv
all the character names are available
movies back to 1936 ?
450 000 characters


# tvtropes.clusters.txt
72 character types drawn from tvtropes.com, along with 501 instances of those types.  The ID field indexes into the Freebase character/actor map ID in character.metadata.tsv.
does not include all the characters available on the dataset 

# name.clusters.txt
970 unique character names used in at least two different movies, along with 2,666 instances of those types.  The ID field indexes into the Freebase character/actor map ID in character.metadata.tsv.


# the paper 
==> read the paper of the movie study !!! 

clustering words to topics => could be used to cluster words to a genre ? 
and try to decipher the genre of a movie by analysing the plot summary

*How has a specific genre evoluted over the years ?*
The dataset contains movies that were filmed over a more-than-60-years time period. Ranging from the 1950s until the 2010s, the dataset usually presents at least 200 movies per year. Most of them present some kind of genre categorisation. 
By leveraging the plot summaries available in the dataset, we could try to analyze how a specific genre (for example horror, or science-fiction) has evolved over the years. How many characters are displayed ? Are the personnas presented similar to each other or is there an evolution trend ? WHat kind of events happen in the plot ? 
This analysis could also try to extract a movie genre from a textual plot summary and fill in the gaps in the dataset. 

add something about technical evolution ? 
coupler ave

_A geographical analysis of Culture_
This subject tackles the question of cultural diffusion and availability. We are constantly in our daily confronted to cultural ideas and concepts. In our globalized world, who creates the cultural content on which a population is confronted ? Is it uniform over the globe or are there disparities ? Using the movie dataset, we can try to dive into a snippet of this broad question. 
Who produces the films ? On what geographical area are the movies available to the public ? Is there a monopoly of a few country (US film) as one would assume on a first-basis or is the movie production more geographically distributed ? 

bias on the language since this db was collected only on English wikipedia pages

_the historical accuracy in movies_
_how is history represented throughout the years_
la mémoire de l'histoire 




Bias of the database, on how the films were collected
characters of the BD => main characters, are they all included ?


Datas historiques sur le cinema, la fréquentation, etc
https://www.data.gouv.fr/fr/organizations/centre-national-du-cinema-et-de-l-image-animee/ 