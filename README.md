# HannibalVector
Vectorising movies based on their IMDb keywords!

## Scraping movies

The files "scrape_movies.py" and "scrape_keywords_for_top_movies.py" are for pulling extra data from IMDb. They accept a start and end year to scrape from and to. Since this repo contains data up to mid-2017, you should only need to run these from 2018 onwards, and only for years after 2016. The keyword scraping only pulls keywords for movies that have been scraped already, so run "scrape movies" before you run "scrape keywords".

## Movie Maths

Check out the Jupyter Notebook. Everything is explained in there. It should all run fine assuming you have Anaconda 4.0 or so, Python 3.5ish, plus I use Seaborn for like, one thing.
