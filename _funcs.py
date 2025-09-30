import argparse
import logging
import json
import os
from io import StringIO, BytesIO
from time import sleep

import requests
import lxml
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.sql import text

import dash
from dash import Dash, html, dash_table, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.express as px
from plotly.graph_objects import Figure

from wordcloud import WordCloud
import pycountry
import ollama


def change_country_name(country_name: str) -> str:
    """Takes country name in `iso alpha_3` format and returns common name of the country"""
    country = pycountry.countries.get(alpha_3=country_name)
    if country:
        if hasattr(country, "common_name"):
            return country.common_name
        else:
            return country.name
    else:
        return country_name


def change_unit_measure_name(unit_measure: str) -> str:
    """Takes measure unit in `world bank` format and returns common name of the unit_measure"""
    MEASURE_UNITS = {
        "PT_LF": "Percent of Labor Force",
        "PC_A": "Annual Percentage Change",
        "PS": "People",
        "IX": "Index",
        "PPP": "Purchasing Power Parity",
        "USD": "USD",
        "PT_B1GQ": "Percent of GDP",
        "PT_GDP": "Percent of GDP",
        "PT": "Percent",
        "0_TO_100": "Gini Index",
        "PA": "Real Interest Rate",
        "10P3PS": "Born per 1,000 people",
        "YR": "Years",
    }
    if unit_measure in MEASURE_UNITS:
        return MEASURE_UNITS[unit_measure]
    else:
        return unit_measure


def world_bank_parser(data_url: str, meta_url: str) -> (pd.DataFrame, dict):
    """Fetch selected database from World Bank Data"""
    # params
    DATABASE_ID, INDICATOR = data_url.split("?")[1].split(
        "&"
    )  # id and name of indicator for world bank urls
    DATABASE_ID = DATABASE_ID.split("=")[-1]
    INDICATOR = INDICATOR.split("=")[-1]
    params = {"DATABASE_ID": DATABASE_ID, "INDICATOR": INDICATOR, "skip": 0}
    json = {"query": f"&$filter=series_description/idno eq '{INDICATOR}'"}

    # metadata
    logging.info("Downloading metadata")
    try:
        r = requests.post(meta_url, json=json)
    except Exception as e:
        logging.error(r.text)
        raise e
    assert r.status_code == 200, logging.error("Unable to load data")
    logging.info("Successfully downloaded metadata")
    name = r.json()["value"][0]["series_description"]["name"]
    description = r.json()["value"][0]["series_description"]["definition_long"]
    metadata = {"name": name, "description": description}

    # data
    logging.info(f'Downloading database "{name}"')
    try:
        r = requests.get(data_url)
    except Exception as e:
        logging.error(r.text)
        raise e
    assert r.status_code == 200, logging.error("Unable to load data")
    df = pd.DataFrame(r.json()["value"])
    n = r.json()["count"] // 1000
    for el in range(n):
        sleep(0.5)
        params["skip"] = 1000 * (el + 1)
        r = requests.get(data_url, params=params)
        assert r.status_code == 200, logging.error("Unable to load data")
        df = pd.concat([df, pd.DataFrame(r.json()["value"])], axis=0)
    df = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE", "UNIT_MEASURE"]]
    possible_countries = [c.alpha_3 for c in pycountry.countries]
    df = df[df["REF_AREA"].isin(possible_countries)]
    df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(np.int16)
    df["OBS_VALUE"] = df["OBS_VALUE"].astype(np.float32)
    df["REF_AREA"] = df["REF_AREA"].apply(change_country_name)
    df["UNIT_MEASURE"] = df["UNIT_MEASURE"].apply(change_unit_measure_name)
    df.columns = [
        "country",
        "year",
        f"{df['UNIT_MEASURE'].value_counts().sort_values(ascending=False).head(1).index[0]}",
        "unit_measure",
    ]
    df.drop(columns=["unit_measure"], inplace=True)
    df = df.sort_values(by="year")
    logging.info(f'Successfully downloaded database "{name}"')
    return (df, metadata)


def oecd_parser(data_url: str, meta_url: str) -> (pd.DataFrame, dict):
    """Fetch selected database from OECD Data"""
    # metadata
    logging.info("Downloading metadata")
    try:
        r = requests.get(meta_url)
    except Exception as e:
        logging.error(r.text)
        raise e
    assert r.status_code == 200, logging.error("Unable to load data")
    logging.info("Successfully downloaded metadata")
    soup = BeautifulSoup(r.text, features="xml")
    if soup is not None:
        description = soup.find("common:Description", {"xml:lang": "en"}).text.strip()
        name = soup.find("common:Name", {"xml:lang": "en"}).text.strip()
        metadata = {"name": name, "description": description}
    else:
        logging.error("Unable to parse data")
        raise Exception

    # data
    logging.info(f'Downloading database "{name}"')
    try:
        r = requests.get(data_url)
    except Exception as e:
        logging.error(r.text)
        raise e
    assert r.status_code == 200, logging.error("Unable to load data")
    df = pd.read_csv(StringIO(r.text))
    df = df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE", "UNIT_MEASURE"]]
    possible_countries = [c.alpha_3 for c in pycountry.countries]
    df = df[df["REF_AREA"].isin(possible_countries)]
    df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(np.int16)
    df["OBS_VALUE"] = df["OBS_VALUE"].astype(np.float32)
    df["REF_AREA"] = df["REF_AREA"].apply(change_country_name)
    df["UNIT_MEASURE"] = df["UNIT_MEASURE"].apply(change_unit_measure_name)
    df.columns = [
        "country",
        "year",
        f"{df['UNIT_MEASURE'].value_counts().sort_values(ascending=False).head(1).index[0]}",
        "unit_measure",
    ]
    df.drop(columns=["unit_measure"], inplace=True)
    df = df.sort_values(by="year")
    logging.info(f'Successfully downloaded database "{name}"')
    return (df, metadata)


def get_country_flag(country: str) -> str:
    """Downloads the flag of country via `restcounties` api and returns path of the downloaded flag image"""
    flag_path = os.path.join("assets", f"{country}.png")
    logging.info("Checking if flag was downloaded")
    if os.path.isfile(flag_path):
        logging.info("Flag was already downloaded")
        return flag_path
    else:
        logging.info("Flag was not downloaded. Start downloading flag")
        r = requests.get(f"https://restcountries.com/v3.1/name/{country}")
        if r.status_code == 200:
            flag_url = r.json()[0]["flags"]["png"]
            flag = requests.get(flag_url)
            with open(flag_path, mode="wb") as file:
                file.write(flag.content)
            logging.info("Successfully downloaded flag")
            return flag_path
        else:
            logging.info("Flag not found")
            DEFAULT_IMAGE_PATH = "assets/image.png"
            return DEFAULT_IMAGE_PATH


def postgres_save_data(
    df: pd.DataFrame, table_name: str, username: str, password: str, port: str | int
) -> None:
    """Saves data to postgres database"""
    assert isinstance(table_name, str), logging.error("Name of table should be string")
    assert isinstance(username, str), logging.error("Username should be string")
    assert isinstance(password, str), logging.error("Password should be string")
    assert isinstance(port, str | int), logging.error(
        "Port should be integer or string"
    )
    port = int(port)

    logging.info(f"Uploading data to table {table_name}")
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{username}:{password}@localhost:{port}"
        )
    except Exception as e:
        logging.error(e)
        raise e
    with engine.connect() as connection:
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
    logging.info(f"Successfully uploaded data to table {table_name}")


def csv_save_data(df: pd.DataFrame, table_name: str, path: str) -> None:
    """Saves data to csv file"""
    abs_path = os.path.join(path, table_name)
    df.to_csv(f"{abs_path}.csv", index=False)


def postgres_read_data(
    table_name: str, username: str, password: str, port: str | int
) -> pd.DataFrame:
    """Reads data from postgres database"""
    assert isinstance(username, str), logging.error("Username should be string")
    assert isinstance(password, str), logging.error("Password should be string")
    assert isinstance(port, str | int), logging.error(
        "Port should be integer or string"
    )
    port = int(port)

    try:
        engine = create_engine(
            f"postgresql+psycopg2://{username}:{password}@localhost:{port}"
        )
    except Exception as e:
        logging.error(e)
        raise e
    with engine.connect() as connection:
        df = connection.execute(text(f"SELECT * FROM {table_name}")).fetchall()
        df = pd.DataFrame(df)
    return df


def csv_read_data(table_name: str, path: str) -> pd.DataFrame:
    """Reads data from csv file"""
    abs_path = os.path.join(path, f"{table_name}.csv")
    df = pd.read_csv(abs_path)
    return df


def get_postgres_table_names(username: str, password: str, port: str | int) -> list:
    """Get list of table names from postgres database and filters them (select only table names for the `Macro dashboard` application)"""
    assert isinstance(username, str), logging.error("Username should be string")
    assert isinstance(password, str), logging.error("Password should be string")
    assert isinstance(port, str | int), logging.error(
        "Port should be integer or string"
    )
    port = int(port)

    try:
        engine = create_engine(
            f"postgresql+psycopg2://{username}:{password}@localhost:{port}"
        )
    except Exception as e:
        logging.error(e)
        raise e
    with engine.connect() as connection:
        df = connection.execute(
            text(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema' AND tablename LIKE 'MD%'"
            )
        ).fetchall()
        df = pd.DataFrame(df)["tablename"]
    return df.to_list()


def get_csv_table_names(path: str) -> list:
    """Get list of table names from csv database"""
    tables = list(filter(lambda x: x.endswith(".csv"), os.listdir(path)))
    tables = list(map(lambda x: x.strip(".csv"), tables))
    return tables


def save_metadata(data: dict, path: str) -> None:
    """Saves datasets metadata in json file"""
    assert isinstance(data, dict), logging.error(
        "Metadata should be in the format of dictionary"
    )
    full_path = os.path.join(path, "METADATA.json")
    with open(full_path, mode="w", encoding="utf-8") as file:
        json.dump(data, file)
    logging.info(f"Successufully saved metadata at {full_path}")


def read_metadata(path: str) -> dict:
    """Reads metadata file"""
    assert os.path.isdir("data"), logging.error("Invalid path for metadata")
    full_path = os.path.join(path, "METADATA.json")
    metadata = json.load(open(full_path))
    return metadata


def ollama_country(model: str, country: str) -> str:
    """Creates prompt with description of the country's economical situation for `ollama` llm via `ollama` api"""
    logging.info("Start generating llm response")
    PROMPT = f"Give short (about 100 words) description of economic development in {country} in official report style with emojis. Do not use numbers."
    response = ollama.generate(model, PROMPT, keep_alive=False)
    logging.info("Successfully generated llm response")
    assert response["done"], logging.error("Unable to connect to ollama llm")
    return response["response"]


def ollama_world(model: str) -> str:
    """Creates prompt with description of the whole world economical situation"""
    logging.info("Start generating llm response")
    PROMPT = "Give short (about 100 words) description of economic development in the world today in official report style with emojis. Do not use numbers."
    response = ollama.generate(model, PROMPT, keep_alive=False)
    logging.info("Successfully generated llm response")
    assert response["done"], logging.error("Unable to connect to ollama llm")
    return response["response"]


def ollama_word_cloud(model: str) -> None:
    """Creates prompt with set of words for creating word cloud"""
    logging.info("Start generating llm response")
    PROMPT = "Give set of random words (about 200 words - words may repeat) that describe current economical situation in the world. Do not use numbers."
    response = ollama.generate(model, PROMPT, keep_alive=False)
    logging.info("Successfully generated llm response")
    assert response["done"], logging.error("Unable to connect to ollama llm")
    cloud = WordCloud(width=800, height=400, background_color="black", colormap="BuPu")
    cloud.generate(response["response"])
    plt.figure()
    plt.imshow(cloud)
    plt.axis("off")
    plt.savefig("assets/ollama_cloud.png", bbox_inches="tight", pad_inches=0)


def text_box(content: str, center=None) -> html.Div:
    """Takes text content and returns `html.div` with text styled in the box"""
    result = html.Div(
        children=content,
        style={
            "border": "1px solid #ddd",
            "padding": "10px",
            "margin": "10px",
            "border-radius": "5px",
            "textAlign": center,
        },
    )
    return result


def graph_box(figure: Figure, description: str) -> html.Div:
    """Takes figure and returns `html.div` with graph and its description styled in the box"""
    if description is None:
        description = ""
    result = html.Div(
        children=[dcc.Graph(figure=figure), text_box(description)],
        style={
            "border": "1px solid #ddd",
            "padding": "10px",
            "margin": "10px",
            "border-radius": "5px",
        },
    )
    return result


def markdown_box(content: str) -> html.Div:
    """Takes markdown content and returns `html.div` with markdown text styled in the box"""
    result = html.Div(
        children=dcc.Markdown(content, mathjax=True),
        style={
            "border": "1px solid #ddd",
            "padding": "10px",
            "margin": "10px",
            "border-radius": "5px",
        },
    )
    return result


def image_box(image_path: str) -> html.Div:
    """Takes path to the image and returns `html.div` with the image styled in the box"""
    result = html.Div(
        html.Img(src=image_path),
        style={
            "border": "1px solid #ddd",
            "padding": "10px",
            "margin": "10px",
            "borderRadius": "5px",
            "textAlign": "center",
            "width": "fit-content",
        },
    )
    return result


def generate_map(data: pd.DataFrame, title: str) -> Figure:
    """Generates plotly map"""
    unit_measure = data.columns[-1]
    fig = px.choropleth(
        data,
        locations="country",
        locationmode="country names",
        color=unit_measure,
        hover_name="country",
        hover_data=unit_measure,
        title=title,
        animation_frame="year",
    )
    fig.update_geos(projection_type="natural earth")
    return fig


def generate_histplot(data: pd.DataFrame, title: str) -> Figure:
    """Generates plotly histogram"""
    unit_measure = data.columns[-1]
    fig = px.histogram(
        data, marginal="violin", x=unit_measure, title=title, animation_frame="year"
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    return fig


def generate_lineplot(data: pd.DataFrame, title: str, country: str) -> Figure:
    """Generates plotly lineplot"""
    data = data[data["country"] == country]
    unit_measure = data.columns[-1]
    fig = px.line(data, x="year", y=unit_measure, title=title)
    return fig


def generate_boxplot(data: pd.DataFrame, title: str) -> Figure:
    """Generates plotly boxplot"""
    unit_measure = data.columns[-1]
    fig = px.box(data, x=unit_measure, title=title, animation_frame="year")
    return fig


def generate_violinplot(data: pd.DataFrame, title: str) -> Figure:
    """Generates plotly violinplot"""
    unit_measure = data.columns[-1]
    fig = px.violin(data, x=unit_measure, title=title, animation_frame="year")
    return fig


def generate_scatterplot(
    data: pd.DataFrame, x: str, y: str, title: str, country=None
) -> Figure:
    """Generates plotly scatterplot"""
    unit_measure = data.columns[-1]
    if country is not None:
        data = data[data["country"] == country]
        fig = px.scatter(
            data,
            x=x,
            y=y,
            title=title,
            hover_data="year",
            trendline="ols",
            trendline_color_override="purple",
        )
    elif country is None:
        fig = px.scatter(
            data,
            x=x,
            y=y,
            title=title,
            animation_frame="year",
            hover_data="country",
            trendline="ols",
            trendline_color_override="purple",
        )
    return fig


def web_app(CONFIG: dict, METADATA: dict, port: int) -> None:
    """Function for creating plotly dashboard with web interface"""
    STYLE = {
        "border": "1px solid #ddd",
        "padding": "10px",
        "margin": "10px",
        "border-radius": "5px",
    }

    if CONFIG["USE_OLLAMA"]:
        ollama_model = CONFIG["OLLAMA_MODEL"]
        world_description = ollama_world(ollama_model)
        ollama_word_cloud(ollama_model)

    if CONFIG["USE_POSTGRES"]:
        args = (
            CONFIG["POSTGRES_USERNAME"],
            CONFIG["POSTGRES_PASSWORD"],
            CONFIG["POSTGRES_PORT"],
        )
        df_gdp = postgres_read_data('"MD GDP"', *args)
        df_gdp_change = postgres_read_data('"MD GDP_CHANGE"', *args)
        df_gdp_pc = postgres_read_data('"MD GDP_PER_CAPITA"', *args)
        df_gdp_pc_change = postgres_read_data('"MD GDP_PER_CAPITA_CHANGE"', *args)
        df_gni = postgres_read_data('"MD GNI"', *args)
        df_gni_change = postgres_read_data('"MD GNI_CHANGE"', *args)
        df_gni_pc = postgres_read_data('"MD GNI_PER_CAPITA"', *args)
        df_gni_pc_change = postgres_read_data('"MD GNI_PER_CAPITA_CHANGE"', *args)
        df_inflation = postgres_read_data('"MD INFLATION"', *args)
        df_real_rate = postgres_read_data('"MD REAL_RATE"', *args)
        df_population = postgres_read_data('"MD POPULATION"', *args)
        df_population_change = (postgres_read_data('"MD POPULATION_CHANGE"', *args),)
        df_birth_rate = postgres_read_data('"MD BIRTH_RATE"', *args)
        df_unemployment = postgres_read_data('"MD UNEMPLOYMENT"', *args)
        df_government_budget_balance = postgres_read_data(
            '"MD GOVERNMENT_BUDGET_BALANCE"', *args
        )
        df_export_gdp = postgres_read_data('"MD EXPORT_PERCENT_GDP"', *args)
        df_import_gdp = postgres_read_data('"MD IMPORT_PERCENT_GDP"', *args)
        df_literacy = postgres_read_data('"MD LITERACY"', *args)
        df_gini = postgres_read_data('"MD GINI"', *args)
        df_gross_debt_gdp = postgres_read_data('"MD GROSS_DEBT_GDP"', *args)
        df_labour_participation_rate = postgres_read_data(
            '"MD LABOUR_PARTICIPATION_RATE"', *args
        )
        df_gdp_on_rd = postgres_read_data('"MD GDP_PERCENT_ON_RD"', *args)
        df_gdp_on_health = postgres_read_data('"MD GDP_PERCENT_ON_HEALTH"', *args)
        df_expected_lifetime = postgres_read_data('"MD EXPECTED_LIFETIME"', *args)
        df_phillips = pd.merge(
            df_unemployment, df_inflation, on=("country", "year"), how="outer"
        ).dropna()
        df_export_import_gdp = pd.merge(
            df_export_gdp,
            df_import_gdp,
            on=("country", "year"),
            how="outer",
            suffixes=["_x", "_y"],
        ).dropna()

    elif not CONFIG["USE_POSTGRES"]:
        args = ("data",)
        df_gdp = csv_read_data("MD GDP", *args)
        df_gdp_change = csv_read_data("MD GDP_CHANGE", *args)
        df_gdp_pc = csv_read_data("MD GDP_PER_CAPITA", *args)
        df_gdp_pc_change = csv_read_data("MD GDP_PER_CAPITA_CHANGE", *args)
        df_gni = csv_read_data("MD GNI", *args)
        df_gni_change = csv_read_data("MD GNI_CHANGE", *args)
        df_gni_pc = csv_read_data("MD GNI_PER_CAPITA", *args)
        df_gni_pc_change = csv_read_data("MD GNI_PER_CAPITA_CHANGE", *args)
        df_inflation = csv_read_data("MD INFLATION", *args)
        df_real_rate = csv_read_data("MD REAL_RATE", *args)
        df_population = csv_read_data("MD POPULATION", *args)
        df_population_change = csv_read_data("MD POPULATION_CHANGE", *args)
        df_birth_rate = csv_read_data("MD BIRTH_RATE", *args)
        df_unemployment = csv_read_data("MD UNEMPLOYMENT", *args)
        df_government_budget_balance = csv_read_data(
            "MD GOVERNMENT_BUDGET_BALANCE", *args
        )
        df_export_gdp = csv_read_data("MD EXPORT_PERCENT_GDP", *args)
        df_import_gdp = csv_read_data("MD IMPORT_PERCENT_GDP", *args)
        df_literacy = csv_read_data("MD LITERACY", *args)
        df_gini = csv_read_data("MD GINI", *args)
        df_gross_debt_gdp = csv_read_data("MD GROSS_DEBT_GDP", *args)
        df_labour_participation_rate = csv_read_data(
            "MD LABOUR_PARTICIPATION_RATE", *args
        )
        df_gdp_on_rd = csv_read_data("MD GDP_PERCENT_ON_RD", *args)
        df_gdp_on_health = csv_read_data("MD GDP_PERCENT_ON_HEALTH", *args)
        df_expected_lifetime = csv_read_data("MD EXPECTED_LIFETIME", *args)
        df_phillips = pd.merge(
            df_unemployment, df_inflation, on=("country", "year"), how="outer"
        ).dropna()
        df_export_import_gdp = pd.merge(
            df_export_gdp,
            df_import_gdp,
            on=("country", "year"),
            how="outer",
            suffixes=["_x", "_y"],
        ).dropna()

    datasets_names = list(map(lambda x: x.lstrip("MD "), METADATA.keys()))
    datasets = [
        df_gdp,
        df_gdp_change,
        df_gdp_pc,
        df_gdp_pc_change,
        df_gni,
        df_gni_change,
        df_gni_pc,
        df_gni_pc_change,
        df_inflation,
        df_real_rate,
        df_population,
        df_population_change,
        df_birth_rate,
        df_unemployment,
        df_government_budget_balance,
        df_export_gdp,
        df_import_gdp,
        df_literacy,
        df_gini,
        df_gross_debt_gdp,
        df_labour_participation_rate,
        df_expected_lifetime,
        df_gdp_on_rd,
        df_gdp_on_health,
    ]
    countries = df_gdp["country"].sort_values().unique()
    data_dict = {i: j for i, j in zip(datasets_names, datasets)}
    country_list = df_gdp["country"].sort_values().unique()
    pio.templates.default = "plotly_dark"
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        suppress_callback_exceptions=True,
    )
    app.title = "Macroeconomic dashboard"

    app.layout = html.Div(
        [
            html.H1("Macroeconomic dashboard", style={"textAlign": "center"}),
            text_box("World economics overview", center="center"),
            html.Div(
                dcc.Dropdown(
                    id="table-selector",
                    options=list(data_dict.keys()),
                    value=list(data_dict.keys())[0],
                    placeholder="Select a metric",
                    style={"textAlign": "center"},
                ),
                style=STYLE,
            ),
            html.Div(dash_table.DataTable(id="data-table", page_size=10), style=STYLE),
            (
                dbc.Row(
                    [
                        dbc.Col(markdown_box(world_description), align="center"),
                        dbc.Col(image_box("assets/ollama_cloud.png"), align="center"),
                    ]
                )
                if CONFIG["USE_OLLAMA"]
                else dbc.Row([dbc.Col(), dbc.Col()])
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_gdp_change,
                                METADATA["MD GDP_CHANGE"]["official_name"],
                            ),
                            METADATA["MD GDP_CHANGE"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_gni_change,
                                METADATA["MD GNI_CHANGE"]["official_name"],
                            ),
                            METADATA["MD GNI_CHANGE"]["description"],
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_map(df_gdp, METADATA["MD GDP"]["official_name"]),
                            METADATA["MD GDP"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_map(df_gni, METADATA["MD GNI"]["official_name"]),
                            METADATA["MD GNI"]["description"],
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_gdp_pc,
                                METADATA["MD GDP_PER_CAPITA"]["official_name"],
                            ),
                            METADATA["MD GDP_PER_CAPITA"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_gni_pc,
                                METADATA["MD GNI_PER_CAPITA"]["official_name"],
                            ),
                            METADATA["MD GNI_PER_CAPITA"]["description"],
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_gdp_pc_change,
                                METADATA["MD GDP_PER_CAPITA_CHANGE"]["official_name"],
                            ),
                            METADATA["MD GDP_PER_CAPITA_CHANGE"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_gni_pc_change,
                                METADATA["MD GNI_PER_CAPITA_CHANGE"]["official_name"],
                            ),
                            METADATA["MD GNI_PER_CAPITA_CHANGE"]["description"],
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_histplot(
                                df_gdp_change,
                                METADATA["MD GDP_CHANGE"]["official_name"],
                            ),
                            METADATA["MD GDP_CHANGE"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_violinplot(
                                df_gdp_pc_change,
                                METADATA["MD GDP_PER_CAPITA_CHANGE"]["official_name"],
                            ),
                            METADATA["MD GDP_PER_CAPITA_CHANGE"]["description"],
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_real_rate, METADATA["MD REAL_RATE"]["official_name"]
                            ),
                            METADATA["MD REAL_RATE"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_histplot(
                                df_real_rate, METADATA["MD REAL_RATE"]["official_name"]
                            ),
                            METADATA["MD REAL_RATE"]["description"],
                        )
                    ),
                ]
            ),
            html.Div(
                graph_box(
                    generate_map(
                        df_birth_rate, METADATA["MD BIRTH_RATE"]["official_name"]
                    ),
                    METADATA["MD BIRTH_RATE"]["description"],
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_inflation, METADATA["MD INFLATION"]["official_name"]
                            ),
                            METADATA["MD INFLATION"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_unemployment,
                                METADATA["MD UNEMPLOYMENT"]["official_name"],
                            ),
                            METADATA["MD UNEMPLOYMENT"]["description"],
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_histplot(
                                df_labour_participation_rate,
                                METADATA["MD LABOUR_PARTICIPATION_RATE"][
                                    "official_name"
                                ],
                            ),
                            METADATA["MD LABOUR_PARTICIPATION_RATE"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_labour_participation_rate,
                                METADATA["MD LABOUR_PARTICIPATION_RATE"][
                                    "official_name"
                                ],
                            ),
                            METADATA["MD LABOUR_PARTICIPATION_RATE"]["description"],
                        )
                    ),
                ]
            ),
            html.Div(
                graph_box(
                    generate_scatterplot(
                        df_phillips,
                        "Percent of Labor Force",
                        "Annual Percentage Change",
                        "Phillips Curve",
                    ),
                    "Inflation & Unemployment for different countries",
                )
            ),
            html.Div(
                graph_box(
                    generate_boxplot(
                        df_government_budget_balance,
                        METADATA["MD GOVERNMENT_BUDGET_BALANCE"]["official_name"],
                    ),
                    METADATA["MD GOVERNMENT_BUDGET_BALANCE"]["description"],
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_export_gdp,
                                METADATA["MD EXPORT_PERCENT_GDP"]["official_name"],
                            ),
                            METADATA["MD EXPORT_PERCENT_GDP"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_map(
                                df_import_gdp,
                                METADATA["MD IMPORT_PERCENT_GDP"]["official_name"],
                            ),
                            METADATA["MD IMPORT_PERCENT_GDP"]["description"],
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_map(df_gini, METADATA["MD GINI"]["official_name"]),
                            METADATA["MD GINI"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_violinplot(
                                df_literacy, METADATA["MD LITERACY"]["official_name"]
                            ),
                            METADATA["MD LITERACY"]["description"],
                        )
                    ),
                ]
            ),
            html.Div(
                graph_box(
                    generate_map(
                        df_gross_debt_gdp,
                        METADATA["MD GROSS_DEBT_GDP"]["official_name"],
                    ),
                    METADATA["MD GROSS_DEBT_GDP"]["description"],
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_box(
                            generate_boxplot(
                                df_gdp_on_rd,
                                METADATA["MD GDP_PERCENT_ON_RD"]["official_name"],
                            ),
                            METADATA["MD GDP_PERCENT_ON_RD"]["description"],
                        )
                    ),
                    dbc.Col(
                        graph_box(
                            generate_violinplot(
                                df_gdp_on_health,
                                METADATA["MD GDP_PERCENT_ON_HEALTH"]["official_name"],
                            ),
                            METADATA["MD GDP_PERCENT_ON_HEALTH"]["description"],
                        )
                    ),
                ],
            ),
            html.Div(
                graph_box(
                    generate_map(
                        df_expected_lifetime,
                        METADATA["MD EXPECTED_LIFETIME"]["official_name"],
                    ),
                    METADATA["MD EXPECTED_LIFETIME"]["description"],
                )
            ),
            text_box("Country economics overview", center="center"),
            html.Div(
                dcc.Dropdown(
                    id="dropdown-selector",
                    options=country_list,
                    value="United States",
                    placeholder="Select a country",
                    style={"textAlign": "center"},
                ),
                style=STYLE,
            ),
            dcc.Store(id="selected-country-store"),
            (
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Markdown(
                                id="ollama-description-text", mathjax=True, style=STYLE
                            ),
                            align="center",
                        ),
                        dbc.Col(
                            html.Img(id="selected-country-flag", style=STYLE),
                            align="center",
                        ),
                    ]
                )
                if CONFIG["USE_OLLAMA"]
                else dbc.Row([dbc.Col(), dbc.Col()])
            ),
            html.Div(dcc.Graph(id="selected-country-map"), style=STYLE),
            dbc.Row(
                [
                    dbc.Col(html.Div(dcc.Graph(id="gdp-graph"), style=STYLE)),
                    dbc.Col(html.Div(dcc.Graph(id="gdp_pc-graph"), style=STYLE)),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(dcc.Graph(id="gni-graph"), style=STYLE)),
                    dbc.Col(html.Div(dcc.Graph(id="gni_pc-graph"), style=STYLE)),
                ]
            ),
            html.Div(dcc.Graph(id="population-graph", style=STYLE)),
            dbc.Row(
                [
                    dbc.Col(html.Div(dcc.Graph(id="inflation-graph"), style=STYLE)),
                    dbc.Col(html.Div(dcc.Graph(id="real_rate-graph"), style=STYLE)),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(dcc.Graph(id="birth_rate-graph"), style=STYLE)),
                    dbc.Col(
                        html.Div(dcc.Graph(id="expected_lifetime-graph"), style=STYLE)
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(dcc.Graph(id="unemployment-graph"), style=STYLE)),
                    dbc.Col(
                        html.Div(
                            dcc.Graph(id="labour_participation_rate-graph"), style=STYLE
                        )
                    ),
                ]
            ),
            html.Div(dcc.Graph(id="export_import_gdp-graph"), style=STYLE),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Graph(id="government_budget_balance-graph"), style=STYLE
                        )
                    ),
                    dbc.Col(
                        html.Div(dcc.Graph(id="gross_debt_gdp-graph"), style=STYLE)
                    ),
                ]
            ),
            html.Div(dcc.Graph(id="gini-graph"), style=STYLE),
            html.Div(dcc.Graph(id="gdp_on_health-graph"), style=STYLE),
            text_box("Download Macroeconomics data", center="center"),
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="metric-dropdown",
                                options=datasets_names,
                                value="GDP",
                                clearable=False,
                            )
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="country-dropdown",
                                options=countries,
                                value="United States",
                                clearable=False,
                            )
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="format-dropdown",
                                options=["png", "jpeg", "pdf", "svg", "html"],
                                value="png",
                                clearable=False,
                            )
                        ),
                    ]
                ),
                style=STYLE,
            ),
            html.Div(
                html.Button("Generate and save plot", id="download_button", n_clicks=0),
                style=STYLE | {"textAlign": "center"},
            ),
            html.Div(dcc.Graph(id="download-graph"), style=STYLE),
            dcc.Download(id="download-plot"),
        ]
    )

    @app.callback(
        Output("data-table", "columns"),
        Output("data-table", "data"),
        Input("table-selector", "value"),
    )
    def update_table(selected_metric: str, data_dict=data_dict) -> tuple:
        """Creates table with selected metric"""
        df = data_dict[selected_metric].sort_values(by=["country", "year"])
        columns = [{"name": i, "id": i} for i in df.columns]
        data = df.to_dict("records")
        return columns, data

    @app.callback(
        Output("selected-country-store", "data"), Input("dropdown-selector", "value")
    )
    def update_output(value: str) -> str:
        """Function for storing the selected country via dropdown function"""
        return value

    @app.callback(
        Output("gdp-graph", "figure"),
        Output("gdp_pc-graph", "figure"),
        Output("gni-graph", "figure"),
        Output("gni_pc-graph", "figure"),
        Output("population-graph", "figure"),
        Output("birth_rate-graph", "figure"),
        Output("expected_lifetime-graph", "figure"),
        Output("inflation-graph", "figure"),
        Output("real_rate-graph", "figure"),
        Output("unemployment-graph", "figure"),
        Output("labour_participation_rate-graph", "figure"),
        Output("export_import_gdp-graph", "figure"),
        Output("gini-graph", "figure"),
        Output("government_budget_balance-graph", "figure"),
        Output("gross_debt_gdp-graph", "figure"),
        Output("gdp_on_health-graph", "figure"),
        Input("selected-country-store", "data"),
    )
    def update_graphs(selected_country: str) -> tuple:
        """Function for updating graphs for selected country"""
        fig_gdp = generate_lineplot(
            df_gdp, METADATA["MD GDP"]["official_name"], selected_country
        )
        fig_gdp_pc = generate_lineplot(
            df_gdp_pc, METADATA["MD GDP_PER_CAPITA"]["official_name"], selected_country
        )
        fig_gni = generate_lineplot(
            df_gni, METADATA["MD GNI"]["official_name"], selected_country
        )
        fig_gni_pc = generate_lineplot(
            df_gni_pc, METADATA["MD GNI_PER_CAPITA"]["official_name"], selected_country
        )
        fig_population = generate_lineplot(
            df_population, METADATA["MD POPULATION"]["official_name"], selected_country
        )
        fig_birth_rate = generate_lineplot(
            df_birth_rate, METADATA["MD BIRTH_RATE"]["official_name"], selected_country
        )
        fig_expected_lifetime = generate_lineplot(
            df_expected_lifetime,
            METADATA["MD EXPECTED_LIFETIME"]["official_name"],
            selected_country,
        )
        fig_inflation = generate_lineplot(
            df_inflation, METADATA["MD INFLATION"]["official_name"], selected_country
        )
        fig_real_rate = generate_lineplot(
            df_real_rate, METADATA["MD REAL_RATE"]["official_name"], selected_country
        )
        fig_unemployment = generate_lineplot(
            df_unemployment,
            METADATA["MD UNEMPLOYMENT"]["official_name"],
            selected_country,
        )
        fig_labour_participation_rate = generate_lineplot(
            df_labour_participation_rate,
            METADATA["MD LABOUR_PARTICIPATION_RATE"]["official_name"],
            selected_country,
        )
        fig_export_import_gdp = generate_scatterplot(
            df_export_import_gdp,
            "Percent of GDP_x",
            "Percent of GDP_y",
            "Export-Import scatter plot",
            selected_country,
        )
        fig_export_import_gdp.update_layout(
            xaxis_title="Export, % of GDP", yaxis_title="Import, % of GDP"
        )
        fig_gini = generate_lineplot(
            df_gini, METADATA["MD GINI"]["official_name"], selected_country
        )
        fig_government_budget_balance = generate_lineplot(
            df_government_budget_balance,
            METADATA["MD GOVERNMENT_BUDGET_BALANCE"]["official_name"],
            selected_country,
        )
        fig_gross_debt_gdp = generate_lineplot(
            df_gross_debt_gdp,
            METADATA["MD GROSS_DEBT_GDP"]["official_name"],
            selected_country,
        )
        fig_gdp_on_health = generate_lineplot(
            df_gdp_on_health,
            METADATA["MD GDP_PERCENT_ON_HEALTH"]["official_name"],
            selected_country,
        )
        return (
            fig_gdp,
            fig_gdp_pc,
            fig_gni,
            fig_gni_pc,
            fig_population,
            fig_birth_rate,
            fig_expected_lifetime,
            fig_inflation,
            fig_real_rate,
            fig_unemployment,
            fig_labour_participation_rate,
            fig_export_import_gdp,
            fig_gini,
            fig_government_budget_balance,
            fig_gross_debt_gdp,
            fig_gdp_on_health,
        )

    @app.callback(
        Output("ollama-description-text", "children"),
        Input("selected-country-store", "data"),
    )
    def ollama_country_description(selected_country: str) -> str:
        """Function for generating llm description of economics of the selected country"""
        if selected_country is not None:
            ollama_description = ollama_country(ollama_model, selected_country)
            return ollama_description

    @app.callback(
        Output("selected-country-flag", "src"), Input("selected-country-store", "data")
    )
    def get_flag(selected_country: str) -> str:
        """Function for updating the flag of the selected country"""
        if selected_country is not None:
            flag_path = get_country_flag(selected_country)
            return flag_path

    @app.callback(
        Output("selected-country-map", "figure"),
        Input("selected-country-store", "data"),
    )
    def country_on_map(selected_country: str) -> Figure:
        """Function for generating map with the selected country"""
        if selected_country is not None:
            fig = px.choropleth(
                locations=[selected_country],
                locationmode="country names",
                title="World Map",
            )
            fig.update_traces(marker_line_width=1, marker_line_color="white")
            return fig

    @app.callback(
        Output("download-graph", "figure"),
        Input("metric-dropdown", "value"),
        Input("country-dropdown", "value"),
    )
    def update_graph_before_saving(
        selected_metric: str, selected_country: str
    ) -> Figure:
        """Function for generating graph before downloading"""
        df = data_dict[selected_metric]
        fig = generate_lineplot(
            df, f"Graph with {selected_metric} of {selected_country}", selected_country
        )
        return fig

    @app.callback(
        Output("download-plot", "data"),
        Input("download_button", "n_clicks"),
        State("metric-dropdown", "value"),
        State("country-dropdown", "value"),
        State("format-dropdown", "value"),
        prevent_initial_call=True,
    )
    def save_graph(
        n_clicks: int, selected_metric: str, selected_country: str, output_format: str
    ) -> "File":
        """Downloads the plot for selected country with selected metric in the preferred format"""
        if n_clicks > 0:
            logging.info("Start downloading plot")
            df = data_dict[selected_metric]
            filtered_df = df[df["country"] == selected_country]
            buf = BytesIO()

            if output_format != "html":
                plt.style.use("dark_background")
                fig = plt.figure(figsize=(12, 5))
                x_axis = filtered_df["year"]
                y_axis = filtered_df[filtered_df.columns[-1]]
                ax = fig.add_subplot(111)
                ax.grid(True, alpha=0.5)
                ax.plot(x_axis, y_axis, color="blue", marker=".")
                ax.set_title(f"{selected_metric} of {selected_country}", fontsize=14)
                ax.set_xlabel("Year", fontsize=12)
                ax.set_ylabel(f"{selected_metric}", fontsize=12)
                fig.savefig(buf, format=output_format, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                logging.info("Successfully downloaded plot")
                return dcc.send_bytes(
                    buf.getvalue(),
                    filename=f"{selected_country} - {selected_metric}.{output_format}",
                )

            elif output_format == "html":
                fig = generate_lineplot(
                    df, f"{selected_metric} of {selected_country}", selected_country
                )
                buf.write(fig.to_html(buf).encode("utf-8"))
                buf.seek(0)
                logging.info("Successfully downloaded plot")
                return dcc.send_bytes(
                    buf.getvalue(),
                    filename=f"{selected_country} - {selected_metric}.{output_format}",
                )

    app.run(port=port)
