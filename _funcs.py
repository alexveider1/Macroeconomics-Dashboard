import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import lxml
import ollama
import pycountry
from time import sleep
import tqdm
import logging
import json
import os
from io import StringIO
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.sql import text

import dash
from dash import Dash, html, dash_table, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.express as px

from wordcloud import WordCloud



def change_country_name(country_name:str)->str:
    country = pycountry.countries.get(alpha_3=country_name)
    if country:
        if hasattr(country, 'common_name'):
            return country.common_name
        else:
            return country.name
    else:
        return country_name



def change_unit_measure_name(unit_measure:str)->str:
    MEASURE_UNITS = {
        'PT_LF': 'Percent of Labor Force',
        'PC_A': 'Annual Percentage Change',
        'PS': 'People',
        'IX': 'Index',
        'PPP': 'Purchasing Power Parity',
        'USD': 'USD',
        'PT_B1GQ': 'Percent of GDP'
    }
    if unit_measure in MEASURE_UNITS:
        return MEASURE_UNITS[unit_measure]
    else:
        return unit_measure



def world_bank_parser(data_url:str, meta_url:str)->(pd.DataFrame, dict):
    """Fetch selected database from World Bank Data"""
    # params
    DATABASE_ID, INDICATOR = data_url.split('?')[1].split('&') # id and name of indicator for world bank urls
    DATABASE_ID = DATABASE_ID.split('=')[-1]
    INDICATOR = INDICATOR.split('=')[-1]
    params = {'DATABASE_ID': DATABASE_ID, 
              'INDICATOR': INDICATOR,
              'skip': 0}
    json = {"query": f"&$filter=series_description/idno eq '{INDICATOR}'"}

    # metadata
    logging.info('Downloading metadata')
    try:
        r = requests.post(meta_url, json=json)
    except Exception as e:
        logging.error(r.text)
        raise e
    assert r.status_code == 200, logging.error('Unable to load data')
    logging.info('Successfully downloaded metadata')
    name = r.json()['value'][0]['series_description']['name']
    description = r.json()['value'][0]['series_description']['definition_long']
    metadata = {
        'name': name,
        'description': description
    }

    # data
    logging.info(f'Downloading database "{name}"')
    try:
        r = requests.get(data_url)
    except Exception as e:
        logging.error(r.text)
        raise e
    assert r.status_code == 200, logging.error('Unable to load data')
    df = pd.DataFrame(r.json()['value'])
    n = r.json()['count'] // 1000
    for el in range(n):
        sleep(0.5)
        params['skip'] = 1000 * (el + 1)
        r = requests.get(data_url, params=params)
        assert r.status_code == 200, logging.error('Unable to load data')
        df = pd.concat([df, pd.DataFrame(r.json()['value'])], axis=0)
    df = df[['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE', 'UNIT_MEASURE']]
    possible_countries = [c.alpha_3 for c in pycountry.countries]
    df = df[df['REF_AREA'].isin(possible_countries)]
    df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(np.int16)
    df['OBS_VALUE'] = df['OBS_VALUE'].astype(np.float32)
    df['REF_AREA'] = df['REF_AREA'].apply(change_country_name)
    df['UNIT_MEASURE'] = df['UNIT_MEASURE'].apply(change_unit_measure_name)
    df.columns = ['country', 'year', f'{df['UNIT_MEASURE'].value_counts().sort_values(ascending=False).head(1).index[0]}', 'unit_measure']
    df.drop(columns=['unit_measure'], inplace=True)
    df = df.sort_values(by='year')
    logging.info(f'Successfully downloaded database "{name}"')
    return (df, metadata)



def oecd_parser(data_url:str, meta_url:str)->(pd.DataFrame, dict):
    """Fetch selected database from OECD Data"""
    # metadata
    logging.info('Downloading metadata')
    try:
        r = requests.get(meta_url)
    except Exception as e:
        logging.error(r.text)
        raise e
    assert r.status_code == 200, logging.error('Unable to load data')
    logging.info('Successfully downloaded metadata')
    soup = BeautifulSoup(r.text, features='xml')
    if soup is not None:
        description = soup.find('common:Description', {'xml:lang': 'en'}).text.strip()
        name = soup.find('common:Name', {'xml:lang': 'en'}).text.strip()
        metadata = {
        'name': name,
        'description': description
        }
    else:
        logging.error('Unable to parse data')
        raise Exception

    # data
    logging.info(f'Downloading database "{name}"')
    try:
        r = requests.get(data_url)
    except Exception as e:
        logging.error(r.text)
        raise e
    assert r.status_code == 200, logging.error('Unable to load data')
    df = pd.read_csv(StringIO(r.text))
    df = df[['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE', 'UNIT_MEASURE']]
    possible_countries = [c.alpha_3 for c in pycountry.countries]
    df = df[df['REF_AREA'].isin(possible_countries)]
    df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(np.int16)
    df['OBS_VALUE'] = df['OBS_VALUE'].astype(np.float32)
    df['REF_AREA'] = df['REF_AREA'].apply(change_country_name)
    df['UNIT_MEASURE'] = df['UNIT_MEASURE'].apply(change_unit_measure_name)
    df.columns = ['country', 'year', f'{df['UNIT_MEASURE'].value_counts().sort_values(ascending=False).head(1).index[0]}', 'unit_measure']
    df.drop(columns=['unit_measure'], inplace=True)
    df = df.sort_values(by='year')
    logging.info(f'Successfully downloaded database "{name}"')
    return (df, metadata)



def get_country_flag(country:str)->str:
    flag_path = os.path.join('assets', f'{country}.png')
    if os.path.isfile(flag_path):
        logging.info('Flag was already downloaded')
        return flag_path
    else: 
        logging.info('Start downloading flag')
        r = requests.get(f'https://restcountries.com/v3.1/name/{country}')
        if r.status_code == 200:
            flag_url = r.json()[0]['flags']['png']
            flag = requests.get(flag_url)
            with open(flag_path, mode='wb') as file:
                file.write(flag.content)
            logging.info('Successfully downloaded flag')
            return flag_path
        else:
            logging.info('Flag not found')
            DEFAULT_IMAGE_PATH = 'assets/image.png'
            return DEFAULT_IMAGE_PATH

    

def postgres_save_data(df:pd.DataFrame, table_name:str, username:str, password:str, port:str|int)->None:
    """Saves data to postgres database"""
    assert isinstance(table_name, str), logging.error('Name of table should be string')
    assert isinstance(username, str), logging.error('Username should be string')
    assert isinstance(password, str), logging.error('Password should be string')
    assert isinstance(port, str|int), logging.error('Port should be integer or string')
    port = int(port)

    logging.info(f'Uploading data to table {table_name}')
    try:
        engine = create_engine(f'postgresql+psycopg2://{username}:{password}@localhost:{port}')
    except Exception as e:
        logging.error(e)
        raise e
    with engine.connect() as connection:
        df.to_sql(
            name=table_name, 
            con=engine, 
            if_exists='replace',
            index=False
        )
    logging.info(f'Successfully uploaded data to table {table_name}')



def csv_save_data(df:pd.DataFrame, table_name:str, path:str)->None:
    """Saves data to csv file"""
    abs_path = os.path.join(path, table_name)
    df.to_csv(f'{abs_path}.csv', index=False)



def postgres_read_data(table_name:str, username:str, password:str, port:str|int)->pd.DataFrame:
    """Reads data from postgres database"""
    assert isinstance(username, str), logging.error('Username should be string')
    assert isinstance(password, str), logging.error('Password should be string')
    assert isinstance(port, str|int), logging.error('Port should be integer or string')
    port = int(port)
    
    try:
        engine = create_engine(f'postgresql+psycopg2://{username}:{password}@localhost:{port}')
    except Exception as e:
        logging.error(e)
        raise e
    with engine.connect() as connection:
        df = connection.execute(text(f'SELECT * FROM {table_name}')).fetchall()
        df = pd.DataFrame(df)
    return df



def csv_read_data(table_name:str, path:str)->pd.DataFrame:
    """Reads data from csv file"""
    abs_path = os.path.join(path, table_name)
    df = pd.read_csv(abs_path)
    return df



def get_postgres_table_names(username:str, password:str, port:str|int)->list:
    """Get list of table names from postgres database and filters them (select only table names for the `Macro dashboard` application)"""
    assert isinstance(username, str), logging.error('Username should be string')
    assert isinstance(password, str), logging.error('Password should be string')
    assert isinstance(port, str|int), logging.error('Port should be integer or string')
    port = int(port)

    try:
        engine = create_engine(f'postgresql+psycopg2://{username}:{password}@localhost:{port}')
    except Exception as e:
        logging.error(e)
        raise e
    with engine.connect() as connection:
        df = connection.execute(
        text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema' AND tablename LIKE 'MD%'")
    ).fetchall()
        df = pd.DataFrame(df)['tablename']
    return df.to_list()



def get_csv_table_names(path:str)->list:
    """Get list of table names from csv database"""
    tables = list(filter(lambda x: x.endswith('.csv'), os.listdir(path)))
    tables = list(map(lambda x: x.strip('.csv'), tables))
    return tables



def save_metadata(data:dict, path:str)->None:
    """Saves datasets metadata in json file"""
    assert isinstance(data, dict), logging.error('Metadata should be in the format of dictionary')
    full_path = os.path.join(path, 'METADATA.json')
    with open(full_path, mode='w', encoding='utf-8') as file:
        json.dump(data, file)
    logging.info(f'Successufully saved metadata at {full_path}')



def read_metadata(path:str)->dict:
    assert os.path.isdir('data'), logging.error('Invalid path for metadata')
    full_path = os.path.join(path, 'METADATA.json')
    metadata = json.load(open(full_path))
    return metadata



def ollama_country(model:str, country:str)->str:
    """Creates prompt with description of the country's economical situation for `ollama` llm via `ollama` api"""
    logging.info('Start generating llm response')
    PROMPT = f'Give short (about 100 words) description of economic development in {country} in official report style with emojis. Do not use numbers.'
    response = ollama.generate(
        model,
        PROMPT,
        keep_alive=False
    )
    logging.info('Successfully generated llm response')
    assert response['done'], logging.error('Unable to connect to ollama llm')
    return response['response']



def ollama_world(model:str)->str:
    """Creates prompt with description of the whole world economical situation"""
    logging.info('Start generating llm response')
    PROMPT = 'Give short (about 100 words) description of economic development in the world today in official report style with emojis. Do not use numbers.'
    response = ollama.generate(
        model,
        PROMPT,
        keep_alive=False
    )
    logging.info('Successfully generated llm response')
    assert response['done'], logging.error('Unable to connect to ollama llm')
    return response['response']


    
def ollama_word_cloud(model:str)->None:
    """Creates prompt with set of words for creating word cloud"""
    logging.info('Start generating llm response')
    PROMPT = 'Give set of random words (about 200 words - words may repeat) that describe current economical situation in the world. Do not use numbers.'
    response = ollama.generate(
        model,
        PROMPT,
        keep_alive=False
    )
    logging.info('Successfully generated llm response')
    assert response['done'], logging.error('Unable to connect to ollama llm')
    cloud = WordCloud(
        width=800, 
        height=400, 
        background_color='black', 
        colormap='BuPu'
    )
    cloud.generate(response['response'])
    plt.figure()
    plt.imshow(cloud)
    plt.axis('off')
    plt.savefig('assets/ollama_cloud.png', bbox_inches='tight', pad_inches=0)



def text_box(content:str, center=None)->html.Div:
    result = html.Div(
        children=content,
        style={
            'border': '1px solid #ddd',
            'padding': '10px',
            'margin': '10px',
            'border-radius': '5px',
            'textAlign': center
        }
    )
    return result



def graph_box(figure, description:str)->html.Div:
    result = html.Div(
        children=[dcc.Graph(figure=figure), text_box(description)],
        style={
            'border': '1px solid #ddd',
            'padding': '10px',
            'margin': '10px',
            'border-radius': '5px'
        }
    )
    return result


    
def markdown_box(content:str)->html.Div:
    result = html.Div(
        children=dcc.Markdown(content, mathjax=True),
        style={
            'border': '1px solid #ddd',
            'padding': '10px',
            'margin': '10px',
            'border-radius': '5px'
        }
    )
    return result



def image_box(image_path:str)->html.Div:
    result = html.Div(
        html.Img(
            src=image_path
        ),
        style={
            'border': '1px solid #ddd',
            'padding': '10px',
            'margin': '10px',
            'borderRadius': '5px',
            'textAlign': 'center',
            'width': 'fit-content'
        }
    )
    return result



def generate_map(data: pd.DataFrame, title:str):
    unit_measure = data.columns[-1]
    fig = px.choropleth(
        data,
        locations = 'country',
        locationmode="country names",
        color=unit_measure,
        hover_name='country',
        hover_data=unit_measure,
        title=title,
        animation_frame='year'
    )
    fig.update_geos(projection_type="natural earth")
    return fig



def generate_histplot(data:pd.DataFrame, title:str):
    unit_measure = data.columns[-1]
    fig = px.histogram(
        data,
        x=unit_measure,
        title=title,
        animation_frame='year'
    )
    fig.update_traces(marker_line_width=1,marker_line_color="white")
    return fig



def generate_lineplot(data:pd.DataFrame, title:str, country:str):
    data = data[data['country']==country]
    unit_measure = data.columns[-1]
    fig = px.line(
        data,
        x='year',
        y=unit_measure,
        title=title
    )
    return fig



def generate_boxplot(data:pd.DataFrame, title:str):
    unit_measure = data.columns[-1]
    fig = px.box(
        data,
        x=unit_measure,
        title=title,
        animation_frame='year'
    )
    return fig


    
def generate_violinplot(data:pd.DataFrame, title:str):
    unit_measure = data.columns[-1]
    fig = px.violin(
        data,
        x=unit_measure,
        title=title,
        animation_frame='year'
    )
    return fig



def web_app(CONFIG:dict, METADATA:dict, port:int)->None:
    STYLE = style={
            'border': '1px solid #ddd',
            'padding': '10px',
            'margin': '10px',
            'border-radius': '5px'
        }
    
    if CONFIG['USE_OLLAMA']:
        ollama_model = CONFIG['OLLAMA_MODEL']
        world_description = ollama_world(ollama_model)
        ollama_word_cloud(ollama_model)

    if CONFIG['USE_POSTGRES']:
        args = CONFIG['POSTGRES_USERNAME'], CONFIG['POSTGRES_PASSWORD'], CONFIG['POSTGRES_PORT']
        df_gdp = postgres_read_data('"MD GDP"', *args)
        df_gdp_change = postgres_read_data('"MD GDP_CHANGE"', *args)
        df_gdp_pc = postgres_read_data('"MD GDP_PER_CAPITA"', *args)
        df_gdp_pc_change = postgres_read_data('"MD GDP_PER_CAPITA_CHANGE"', *args)
        df_inflation = postgres_read_data('"MD INFLATION"', *args)
        df_unemployment = postgres_read_data('"MD UNEMPLOYMENT"', *args)
        df_population = postgres_read_data('"MD POPULATION"', *args)
        df_population_change = postgres_read_data('"MD POPULATION_CHANGE"', *args)
        df_gdp_on_rd = postgres_read_data('"MD GDP_PERCENT_ON_RD"', *args)
        df_gdp_on_health = postgres_read_data('"MD GDP_PERCENT_ON_HEALTH"', *args)
    elif not CONFIG['USE_POSTGRES']:
        args = ['data']
        df_gdp = csv_read_data('MD GDP.csv', *args)
        df_gdp_change = csv_read_data('MD GDP_CHANGE.csv', *args)
        df_gdp_pc = csv_read_data('MD GDP_PER_CAPITA.csv', *args)
        df_gdp_pc_change = csv_read_data('MD GDP_PER_CAPITA_CHANGE.csv', *args)
        df_inflation = csv_read_data('MD INFLATION.csv', *args)
        df_unemployment = csv_read_data('MD UNEMPLOYMENT.csv', *args)
        df_population = csv_read_data('MD POPULATION.csv', *args)
        df_population_change = csv_read_data('MD POPULATION_CHANGE.csv', *args)
        df_gdp_on_rd = csv_read_data('MD GDP_PERCENT_ON_RD.csv', *args)
        df_gdp_on_health = csv_read_data('MD GDP_PERCENT_ON_HEALTH.csv', *args)

    country_list = df_gdp['country'].sort_values().unique()
    pio.templates.default = "plotly_dark"
    app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.title = "Macroeconomic dashboard"
    
    app.layout = html.Div([
            html.H1('Macroeconomic dashboard', style={'textAlign': 'center'}),
            text_box('World economic overview', center='center'),
            dbc.Row([dbc.Col(markdown_box(world_description), align='center'), dbc.Col(image_box('assets/ollama_cloud.png'), align='center')]) if CONFIG['USE_OLLAMA'] else dbc.Row([dbc.Col(), dbc.Col()]), 
            graph_box(generate_map(df_gdp_change, METADATA['MD GDP_CHANGE']['official_name']), METADATA['MD GDP_CHANGE']['description']), 
            graph_box(generate_map(df_gdp, METADATA['MD GDP']['official_name']), METADATA['MD GDP']['description']), 
            graph_box(generate_map(df_gdp_pc, METADATA['MD GDP_PER_CAPITA']['official_name']), METADATA['MD GDP_PER_CAPITA']['description']), 
            graph_box(generate_map(df_gdp_pc_change, METADATA['MD GDP_PER_CAPITA_CHANGE']['official_name']), METADATA['MD GDP_PER_CAPITA_CHANGE']['description']),
            dbc.Row([dbc.Col(graph_box(generate_histplot(df_gdp_change, METADATA['MD GDP_CHANGE']['official_name']), METADATA['MD GDP_CHANGE']['description'])), dbc.Col(graph_box(generate_violinplot(df_gdp_pc_change, METADATA['MD GDP_PER_CAPITA_CHANGE']['official_name']), METADATA['MD GDP_PER_CAPITA_CHANGE']['description']))]), 
            dbc.Row([dbc.Col(graph_box(generate_map(df_inflation, METADATA['MD INFLATION']['official_name']), METADATA['MD INFLATION']['description'])), dbc.Col(graph_box(generate_map(df_unemployment, METADATA['MD UNEMPLOYMENT']['official_name']), METADATA['MD UNEMPLOYMENT']['description']))]),
            dbc.Row([dbc.Col(graph_box(generate_boxplot(df_gdp_on_rd, METADATA['MD GDP_PERCENT_ON_RD']['official_name']), METADATA['MD GDP_PERCENT_ON_RD']['description'])), dbc.Col(graph_box(generate_violinplot(df_gdp_on_health, METADATA['MD GDP_PERCENT_ON_HEALTH']['official_name']), METADATA['MD GDP_PERCENT_ON_HEALTH']['description']))]), 
            text_box('Country economic overview', center='center'), 
            html.Div(dcc.Dropdown(country_list, id='dropdown-selector', placeholder='Select a country', style={'margin':'auto'}, value='United States'), style={'textAlign': 'center'}), 
            dcc.Store(id='selected-country-store'), 
            dbc.Row([dbc.Col(dcc.Markdown(id='ollama-description-text', mathjax=True, style=STYLE), align='center'), dbc.Col(html.Img(id='selected-country-flag', style=STYLE), align='center')]) if CONFIG['USE_OLLAMA'] else dbc.Row([dbc.Col(), dbc.Col()]), 
            dbc.Row([dbc.Col(html.Div(dcc.Graph(id='gdp-graph'), style=STYLE)), dbc.Col(html.Div(dcc.Graph(id='gdp_pc-graph'), style=STYLE))]),
            dcc.Graph(id='population-graph', style=STYLE),
            dbc.Row([dbc.Col(html.Div(dcc.Graph(id='inflation-graph'), style=STYLE)), dbc.Col(html.Div(dcc.Graph(id='unemployment-graph'), style=STYLE))]),
            html.Div(dcc.Graph(id='gdp_on_health-graph'), style=STYLE)
        ])

    @app.callback(
        Output('selected-country-store', 'data'),
        Input('dropdown-selector', 'value')
    )
    def update_output(value):
        return value

    @app.callback(
        Output('gdp-graph', 'figure'), 
        Output('gdp_pc-graph', 'figure'), 
        Output('population-graph', 'figure'), 
        Output('inflation-graph', 'figure'), 
        Output('unemployment-graph', 'figure'), 
        Output('gdp_on_health-graph', 'figure'), 
        Input('selected-country-store', 'data')
    )
    def update_graphs(selected_country):    
        fig_gdp = generate_lineplot(df_gdp, METADATA['MD GDP']['official_name'], selected_country)
        fig_gdp_pc = generate_lineplot(df_gdp_pc, METADATA['MD GDP_PER_CAPITA']['official_name'], selected_country)
        fig_population = generate_lineplot(df_population, METADATA['MD POPULATION']['official_name'], selected_country)
        fig_inflation = generate_lineplot(df_inflation, METADATA['MD INFLATION']['official_name'], selected_country)
        fig_unemployment = generate_lineplot(df_unemployment, METADATA['MD UNEMPLOYMENT']['official_name'], selected_country)
        fig_gdp_on_rd = generate_lineplot(df_gdp_on_rd, METADATA['MD GDP_PERCENT_ON_RD']['official_name'], selected_country)
        fig_gdp_on_health = generate_lineplot(df_gdp_on_health, METADATA['MD GDP_PERCENT_ON_HEALTH']['official_name'], selected_country)
        return fig_gdp, fig_gdp_pc, fig_population, fig_inflation, fig_unemployment, fig_gdp_on_health

    @app.callback(
        Output('ollama-description-text', 'children'), 
        Input('selected-country-store', 'data') 
    )
    def ollama_country_description(selected_country):
        if selected_country is not None:
            ollama_description = ollama_country(ollama_model, selected_country)
            return ollama_description

    @app.callback(
        Output('selected-country-flag', 'src'),
        Input('selected-country-store', 'data')
    )
    def get_flag(selected_country):
        if selected_country is not None:
            flag_path = get_country_flag(selected_country)
            return flag_path
    
    app.run(port=port)
