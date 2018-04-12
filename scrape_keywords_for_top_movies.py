import bs4
import argparse
import urllib.request
import pandas as pd
import logging
import os


PARSER = argparse.ArgumentParser(description='This tool will create csv files of movies scraped from IMDb.com')

PARSER.add_argument('start', help='Year to start scraping from', type=int)
PARSER.add_argument('end', help='Year to start scraping at', type=int)

# TODO: Validate the years (not lower than <1900 or higher than 2100)
ARGS = PARSER.parse_args()

years = range(ARGS.start, ARGS.end + 1)

logging.basicConfig(level=logging.INFO)
logging.info('Scraping keyword data for top movies from %d to %d.' % (ARGS.start, ARGS.end + 1))


def write_to_csv(movies_list, year):
    df = pd.DataFrame(movies_list)
    file_name = 'scraped_movies/keywords_for_top_movies_of_%d.csv' % year
    df.to_csv(file_name, index=False)
    logging.info('Wrote the following file: %s' % file_name)


if __name__ == '__main__':
    for year in years:
        movies_file_name = './scraped_movies/top_movies_of_%d.csv'
        movies = pd.read_csv(movies_file_name % year, encoding = "ISO-8859-1")
        keywords_list = []
        for IMDbId in movies.IMDbId:
            logging.info('Fetching keywords for %s' % IMDbId)
            url = 'http://www.imdb.com/title/%s/keywords' % IMDbId
            with urllib.request.urlopen(url) as response:
                html = response.read()
            soup = bs4.BeautifulSoup(html, "lxml")
            keyword_tags = soup.find_all(attrs={'class': "soda sodavote"})
            keywords = '|'.join([i['data-item-keyword'] for i in keyword_tags])
            movie_data = {'IMDbId': IMDbId,
                          'keywords': keywords}
            keywords_list.append(movie_data)
        write_to_csv(keywords_list, year)
        logging.info('Wrote file for %d' % year)
