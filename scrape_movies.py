import bs4
import argparse
import urllib.request
import pandas as pd
import logging


PARSER = argparse.ArgumentParser(description='This tool will create csv files of movies scraped from IMDb.com')

PARSER.add_argument('start', help='Year to start scraping from', type=int)
PARSER.add_argument('end', help='Year to start scraping at', type=int)

ARGS = PARSER.parse_args()

years = range(ARGS.start, ARGS.end + 1)

logging.basicConfig(level=logging.INFO)
logging.info('Scraping data for top movies from %d to %d.' % (ARGS.start, ARGS.end + 1))


def movie_data_from_soup(movie, year):
    try:
        title_raw = movie.find('h3').text.strip('\n')
        title = title_raw[title_raw.find('\n') + 1:].replace('\n', ' ')
    except:
        title = None
    try:
        genre_raw = movie.find("span", {"class": "genre"}).text.strip('\n')
        genre = genre_raw.strip()
    except:
        genre = None
    try:
        IMDbId = movie.find("div", {"data-caller": "filmosearch"})['data-tconst']
    except:
        IMDbId = None
    try:
        rank_raw = movie.find('span', {'class': "lister-item-index unbold text-primary"}).text
        raw_score = movie.find('div', {'class': "inline-block ratings-imdb-rating"})['data-value']
        rank = int(rank_raw.strip().replace('.', ''))
        IMDb_score = float(raw_score)
    except:
        rank = None
        IMDb_score = None
    movie_data = {'title': title,
                  'genre_list': genre,
                  'box_office_rank': rank,
                  'IMDbId': IMDbId,
                  'IMDb_score': IMDb_score,
                  'release_year': year}
    return movie_data


def build_url(year, page):
    url = 'http://www.imdb.com/search/title?release_date=%d&sort=boxoffice_gross_us,desc&page=%d' % (year, page)
    return url


def write_to_csv(movies_list, year):
    df = pd.DataFrame(movies_list)
    file_name = 'scraped_movies/top_movies_of_%d.csv' % year
    df.to_csv(file_name, index=False)
    logging.info('Wrote the following file: %s' % file_name)


if __name__ == '__main__':
    for year in years:
        movies_list = []
        for page in [1, 2]:
            logging.info('Fetching year %d page %d' % (year, page))
            url = build_url(year, page)
            with urllib.request.urlopen(url) as response:
                html = response.read()
            soup = bs4.BeautifulSoup(html, "lxml")
            movies = soup.findAll("div", {"class": "lister-item mode-advanced"})
            for movie in movies:
                movies_list.append(movie_data_from_soup(movie, year))
        write_to_csv(movies_list, year)
