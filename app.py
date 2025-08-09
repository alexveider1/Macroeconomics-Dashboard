import argparse
import logging
import json
import os
from io import StringIO
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

from _funcs import change_country_name, change_unit_measure_name
from _funcs import world_bank_parser, oecd_parser
from _funcs import postgres_save_data, csv_save_data, postgres_read_data, csv_read_data
from _funcs import get_postgres_table_names, get_csv_table_names
from _funcs import save_metadata, read_metadata
from _funcs import ollama_world, ollama_word_cloud, ollama_country
from _funcs import web_app

import dash
from dash import Dash, html, dash_table, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.express as px
from plotly.graph_objects import Figure

from wordcloud import WordCloud
import pycountry
import ollama


def init() -> tuple:
    """Basic function for the initialization of the application with selected params"""
    # default settings from json-file
    CONFIG = json.load(open("data/CONFIG.json"))
    DEFAULT_METADATA = json.load(open("data/DEFAULT_METADATA.json"))

    # basic argparser
    parser = argparse.ArgumentParser(
        prog="Macro Dashboard",
        description="What the program does: creates a dashboard with graphs for world economies",
        epilog="Illustration of main indicators of different economies",
    )

    # setting from cli (when launching application)
    parser.add_argument(
        "--preload_data",
        default=False,
        help="use preload data or download data from world_bank/oecd via api",
    )
    parser.add_argument(
        "--use_postgres", default=False, help="Use or not postgres for storing data"
    )
    parser.add_argument(
        "--postgres_username",
        type=str,
        default=None,
        help="Username for postgres (required if `USE_POSTGRES`=True)",
    )
    parser.add_argument(
        "--postgres_password",
        type=str,
        default=None,
        help="Password for postgres (required if `USE_POSTGRES`=True)",
    )
    parser.add_argument(
        "--postgres_port",
        type=int,
        default=None,
        help="Port for postgres (required if `USE_POSTGRES`=True)",
    )
    parser.add_argument(
        "--use_ollama",
        default=False,
        help="Use or not ollama llm for description of economic situation",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default=None,
        help="Model name of ollama llm for use",
    )
    parser.add_argument(
        "--web_ui_port",
        type=int,
        default=None,
        help="Port for web application with dashboard",
    )

    args = dict(parser.parse_args().__dict__)

    # replacing default settings with settings changed by user during app launch
    for param in args:
        if args[param] != None:
            CONFIG[param.upper()] = args[param]
    # now use CONFIG for settings of application

    # basic logging both in cli and file 'app.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("data/app.log"), logging.StreamHandler()],
    )
    logging.info("Initialization complete")

    return CONFIG, DEFAULT_METADATA


def prepare_data(CONFIG: dict, METADATA: dict) -> tuple:
    """Basic function for preparing data for the application by downloading it or by asserting that data has been successfully downloaded before (depending on selected application params)"""
    # Preparing datasets and metadata
    ## use postgres for storing databases
    if CONFIG["USE_POSTGRES"]:
        assert isinstance(CONFIG["POSTGRES_USERNAME"], str), logging.error(
            "You should provide `POSTGRES_USERNAME` in order to use postgres as database"
        )
        assert isinstance(CONFIG["POSTGRES_PASSWORD"], str), logging.error(
            "You should provide `POSTGRES_PASSWORD` in order to use postgres as database"
        )
        assert isinstance(CONFIG["POSTGRES_PORT"], str | int), logging.error(
            "You should provide `POSTGRES_PORT` in order to use postgres as database"
        )

        if CONFIG["PRELOAD_DATA"]:
            logging.info("Checking for preloaded data")
            tables = get_postgres_table_names(
                username=CONFIG["POSTGRES_USERNAME"],
                password=CONFIG["POSTGRES_PASSWORD"],
                port=CONFIG["POSTGRES_PORT"],
            )
            METADATA = read_metadata("data")
            assert set(METADATA.keys()).issubset(set(tables)), logging.error(
                "Metadata is not compatible with datasets"
            )
            logging.info("Successfully found preloaded data")

        elif not CONFIG["PRELOAD_DATA"]:
            logging.info("DOWNLOADING WORLD_BANK DATASETS")
            for obj in METADATA.keys():
                if METADATA[obj]["source"] == "WORLD_BANK":
                    db, meta = METADATA[obj]["data_url"], METADATA[obj]["meta_url"]
                    df, metadata = world_bank_parser(db, meta)
                    table_name = obj
                    postgres_save_data(
                        df,
                        table_name=table_name,
                        username=CONFIG["POSTGRES_USERNAME"],
                        password=CONFIG["POSTGRES_PASSWORD"],
                        port=CONFIG["POSTGRES_PORT"],
                    )
                    METADATA[obj]["official_name"] = metadata["name"]
                    METADATA[obj]["description"] = metadata["description"]
                    METADATA[obj]["database"] = "postgres"

            logging.info("Successufully downloaded world_bank datasets")
            save_metadata(METADATA, "data")
            logging.info("Successfully saved metadata")

            logging.info("DOWNLOADING OECD DATASETS")
            for obj in METADATA.keys():
                if METADATA[obj]["source"] == "OECD":
                    db, meta = METADATA[obj]["data_url"], METADATA[obj]["meta_url"]
                    df, metadata = oecd_parser(db, meta)
                    table_name = obj
                    postgres_save_data(
                        df,
                        table_name=table_name,
                        username=CONFIG["POSTGRES_USERNAME"],
                        password=CONFIG["POSTGRES_PASSWORD"],
                        port=CONFIG["POSTGRES_PORT"],
                    )
                    METADATA[obj]["official_name"] = metadata["name"]
                    METADATA[obj]["description"] = metadata["description"]
                    METADATA[obj]["database"] = "postgres"

            logging.info("Successufully downloaded world_bank datasets")
            save_metadata(METADATA, "data")
            logging.info("Successfully saved metadata")

    # use csv files for storing databases
    elif not CONFIG["USE_POSTGRES"]:
        assert os.path.isdir("data"), logging.error("No `data` directory")
        if CONFIG["PRELOAD_DATA"]:
            logging.info("Checking for preloaded data")
            tables = get_csv_table_names("data")
            METADATA = read_metadata("data")
            assert set(METADATA.keys()).issubset(set(tables)), logging.error(
                "Metadata is not compatible with datasets"
            )
            logging.info("Successfully found preloaded data")

        elif not CONFIG["PRELOAD_DATA"]:
            logging.info("Downloading world_bank datasets")
            for obj in METADATA.keys():
                if METADATA[obj]["source"] == "WORLD_BANK":
                    db, meta = METADATA[obj]["data_url"], METADATA[obj]["meta_url"]
                    df, metadata = world_bank_parser(db, meta)
                    table_name = obj
                    csv_save_data(
                        df,
                        table_name=table_name,
                        path=os.path.join(os.getcwd(), "data"),
                    )
                    METADATA[obj]["official_name"] = metadata["name"]
                    METADATA[obj]["description"] = metadata["description"]
                    METADATA[obj]["database"] = "csv"

            logging.info("Successufully downloaded world_bank datasets")
            save_metadata(METADATA, "data")
            logging.info("Successfully saved metadata")

            logging.info("Downloading OECD datasets")
            for obj in METADATA.keys():
                if METADATA[obj]["source"] == "OECD":
                    db, meta = METADATA[obj]["data_url"], METADATA[obj]["meta_url"]
                    df, metadata = oecd_parser(db, meta)
                    table_name = obj
                    csv_save_data(
                        df,
                        table_name=table_name,
                        path=os.path.join(os.getcwd(), "data"),
                    )
                    METADATA[obj]["official_name"] = metadata["name"]
                    METADATA[obj]["description"] = metadata["description"]
                    METADATA[obj]["database"] = "csv"

            logging.info("Successfully downloaded oecd datasets")
            save_metadata(METADATA, "data")
            logging.info("Successfully saved metadata")

    return CONFIG, METADATA


def main(CONFIG: dict, METADATA: dict) -> None:
    """Launches application with chosen params"""
    if CONFIG["USE_OLLAMA"]:
        ollama_model = CONFIG["OLLAMA_MODEL"]
        world_description = ollama_world(ollama_model)
        ollama_word_cloud(ollama_model)
    web_app(CONFIG, METADATA, port=CONFIG["WEB_UI_PORT"])


# launch app
if __name__ == "__main__":
    CONFIG, METADATA = init()
    CONFIG, METADATA = prepare_data(CONFIG, METADATA)
    main(CONFIG, METADATA)
