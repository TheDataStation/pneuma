"""
registrar.py

This module handles dataset registration functionality.
"""
import json
import logging
import os

import duckdb
import fire
import pandas as pd

from pneuma.utils.logging_config import configure_logging
from pneuma.utils.response import Response, ResponseStatus
from pneuma.utils.table_status import TableStatus

configure_logging()
logger = logging.getLogger("Registrar")


class Registrar:
    """
    Registers dataset and its context (metadata) to the database.

    This class provides methods to setup the registration system and to
    register tables & context (metadata).

    ## Attributes
    - **db_path** (`str`): Path to the database file for retrieving content
    summaries & context.
    """
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path

    def setup(self) -> str:
        """
        Setups the database system for registration purposes.

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        try:
            with duckdb.connect(self.db_path) as connection:
                connection.execute("INSTALL httpfs")
                connection.execute("LOAD httpfs")
                logger.info("HTTPFS installed and loaded")

                connection.sql(
                    """CREATE TABLE IF NOT EXISTS table_status (
                        id VARCHAR PRIMARY KEY,
                        table_name VARCHAR NOT NULL,
                        status VARCHAR NOT NULL,
                        time_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        creator VARCHAR NOT NULL,
                        hash VARCHAR NOT NULL,
                        )
                    """
                )
                logger.info("Table `table_status` created")

                # Arbitrary auto-incrementing id for contexts and summaries.
                # Change to "CREATE IF NOT EXISTS" on production.
                connection.sql("CREATE SEQUENCE IF NOT EXISTS id_seq START 1")
                logger.info("ID sequence created")

                # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
                connection.sql(
                    """CREATE TABLE IF NOT EXISTS table_contexts (
                        id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                        table_id VARCHAR NOT NULL REFERENCES table_status(id),
                        context JSON NOT NULL
                        )
                    """
                )
                logger.info("Table `table_contexts` created")

                # DuckDB does not support "ON DELETE CASCADE" so be careful with deletions.
                connection.sql(
                    """CREATE TABLE IF NOT EXISTS table_summaries (
                        id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
                        table_id VARCHAR NOT NULL REFERENCES table_status(id),
                        summary JSON NOT NULL,
                        summary_type VARCHAR NOT NULL,
                        )
                    """
                )
                logger.info("Table `table_summaries` created")

                connection.sql(
                    """CREATE TABLE IF NOT EXISTS indexes (
                        id INTEGER default nextval('id_seq') PRIMARY KEY,
                        name VARCHAR NOT NULL,
                        location VARCHAR NOT NULL,
                        )
                    """
                )
                logger.info("Table `indexes` created")

                connection.sql(
                    """CREATE TABLE IF NOT EXISTS index_table_mappings (
                        index_id INTEGER NOT NULL REFERENCES indexes(id),
                        table_id VARCHAR NOT NULL REFERENCES table_status(id),
                        PRIMARY KEY (index_id, table_id)
                        )
                    """
                )
                logger.info("Table `index_table_mappings` created")

                # TODO: Adjust the response column to the actual response type.
                connection.sql(
                    """CREATE TABLE IF NOT EXISTS query_history (
                        time TIMESTAMP DEFAULT CURRENT_TIMESTAMP PRIMARY KEY,
                        table_id VARCHAR NOT NULL REFERENCES table_status(id), 
                        query VARCHAR NOT NULL,
                        response VARCHAR NOT NULL,
                        querant VARCHAR NOT NULL
                        )
                    """
                )
                logger.info("Table `query_history` created")

                return Response(
                    status=ResponseStatus.SUCCESS,
                    message="Database Initialized.",
                ).to_json()
        except Exception as e:
            return Response(
                status=ResponseStatus.ERROR,
                message=f"Error initializing database: {e}",
            ).to_json()

    def add_tables(
        self,
        path: str,
        creator: str,
        source: str = "file",
        s3_region: str = None,
        s3_access_key: str = None,
        s3_secret_access_key: str = None,
        accept_duplicates: bool = False,
    ) -> str:
        """
        Adds tables into the database.

        ## Args
        - **path** (`str`): The path to a specific table file/folder (`CSV` or
        `parquet`).
        - **creator** (`str`): The creator of the file.
        - **source** (`str`): The dataset source (either `file` or `s3`).
        - **s3_region** (`int`): Amazon S3 region.
        - **s3_access_key** (`int`): Amazon S3 access key.
        - **s3_secret_access_key** (`int`): Amazon S3 secret access key.
        - **accept_duplicates** (`bool`): Option to accept duplicate tables or not.

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        if source not in ["file", "s3"]:
            return "Invalid source. Please use 'file' or 's3'."

        if source == "s3":
            try:
                with duckdb.connect(self.db_path) as connection:
                    connection.execute(
                        f"""
                    SET s3_region='{s3_region}';
                    SET s3_access_key_id='{s3_access_key}';
                    SET s3_secret_access_key='{s3_secret_access_key}';
                    """
                    )
            except Exception as e:
                return Response(
                    status=ResponseStatus.ERROR,
                    message=f"Error connecting to database: {e}",
                ).to_json()

        if os.path.isfile(path):
            return self.__read_table_file(path, creator, accept_duplicates).to_json()
        if os.path.isdir(path):
            return self.__read_table_folder(path, creator, accept_duplicates).to_json()

        return Response(
            status=ResponseStatus.ERROR,
            message=f"Invalid path: {path}",
        ).to_json()

    def add_metadata(
        self, metadata_path: str, table_id: str = ""
    ) -> str:
        """
        Adds metadata into the database.

        ## Args
        - **metadata_path** (`str`): The path to a specific metadata file/folder
        (`TXT` or `CSV`).
        - **table_id** (`str`): A specific table id associated with the metadata.

        ## Returns
        - `str`: A JSON string representing the result of the process (`Response`).
        """
        if os.path.isfile(metadata_path):
            return self.__read_metadata_file(
                metadata_path, table_id
            ).to_json()
        if os.path.isdir(metadata_path):
            return self.__read_metadata_folder(
                metadata_path, table_id
            ).to_json()

        return Response(
            status=ResponseStatus.ERROR,
            message=f"Invalid path: {metadata_path}",
        ).to_json()

    def __read_table_file(
        self,
        path: str,
        creator: str,
        accept_duplicates: bool = False,
    ) -> Response:
        """
        Reads a table file (CSV or Parquet), registers it in the database, and
        updates if an existing table has the same ID.

        ## Args
        - **path** (`str`): The path to a specific table file (`CSV` or `parquet`).
        - **creator** (`str`): The creator of the file.
        - **accept_duplicates** (`bool`): Option to allow duplicate tables or not.

        ## Returns
        - `Response`: A `Response` object of the process.
        """
        try:
            with duckdb.connect(self.db_path) as connection:
                # Index -1 to get the file extension, then slice [1:] to remove the dot.
                file_type = os.path.splitext(path)[-1][1:]

                if file_type not in ["csv", "parquet"]:
                    return Response(
                        status=ResponseStatus.ERROR,
                        message="Invalid file type. Please use 'csv' or 'parquet'.",
                    )

                # If the path contains single quotes, we need to escape them to avoid
                # breaking the SQL query.
                path = path.replace("'", "''")
                if file_type == "csv":
                    name = path.split("/")[-1][:-4]
                    table = connection.sql(
                        f"""SELECT *
                            FROM read_csv(
                                '{path}',
                                auto_detect=True,
                                header=True,
                                ignore_errors=True
                            )"""
                    )
                    table_hash = connection.sql(
                        f"""SELECT md5(string_agg(tbl::text, ''))
                        FROM read_csv(
                            '{path}',
                            auto_detect=True,
                            header=True,
                            ignore_errors=True
                        ) AS tbl"""
                    ).fetchone()[0]
                elif file_type == "parquet":
                    name = path.split("/")[-1][:-8]
                    table = connection.sql(
                        f"""SELECT *
                        FROM read_parquet(
                            '{path}'
                        )"""
                    )
                    table_hash = connection.sql(
                        f"""SELECT md5(string_agg(tbl::text, ''))
                        FROM read_parquet(
                            '{path}'
                        ) AS tbl"""
                    ).fetchone()[0]

                # For ease of keeping track of IDs, we replace backslashes (Windows) with
                # forward slashes (everything else) to make the path (therefore, ID) consistent.
                path = path.replace("\\", "/")

                # We want to avoid double quotes on table names, so we change them to single quotes.
                path = path.replace('"', "''")

                if not accept_duplicates:
                    # Check if table with the same hash already exist
                    table_exist = connection.sql(
                        f"SELECT id FROM table_status WHERE hash = '{table_hash}'"
                    ).fetchone()
                    if table_exist:
                        return Response(
                            status=ResponseStatus.ERROR,
                            message=f"This table already exists in the database with id {table_exist}.",
                        )

                # Check if a table with the same ID already exists
                existing_entry = connection.sql(
                    f"SELECT table_name FROM table_status WHERE id = '{path}'"
                ).fetchone()

                if existing_entry:
                    old_table_name = existing_entry[0]
                    connection.sql(f"DROP TABLE IF EXISTS \"{old_table_name}\"")
                    connection.sql(
                        f"DELETE FROM table_status WHERE id = '{path}'"
                    )

                # The double quote is necessary to consider the path, which may contain
                # full stop that may mess with schema as a single string. Having single quote
                # inside breaks the query, so having the double quote INSIDE the single quote
                # is the only way to make it work.

                # Because we are using double quotes for the table creation, we need to unescape
                # the single quote so the table name fits the ID that is stored in table_status.
                create_path = path.replace("''", "'")
                table.create(f'"{create_path}"')

                connection.sql(
                    f"""INSERT INTO table_status (id, table_name, status, creator, hash)
                    VALUES ('{path}', '{name}', '{TableStatus.REGISTERED}', '{creator}', '{table_hash}')"""
                )
                return Response(
                    status=ResponseStatus.SUCCESS,
                    message=f"Table with ID: {path} has been added to the database.",
                    data={"table_id": path, "table_name": name},
                )
        except Exception as e:
            return Response(
                status=ResponseStatus.ERROR,
                message=f"Error connecting to database: {e}",
            )

    def __read_table_folder(
        self, folder_path: str, creator: str, accept_duplicates: bool = False
    ) -> Response:
        """
        Reads a folder and registers all of its tables to the database.

        ## Args
        - **folder_path** (`str`): The path to a folder containing tables.
        - **creator** (`str`): The creator of the file.
        - **accept_duplicates** (`bool`): Option to allow duplicate tables or not.

        ## Returns
        - `Response`: A `Response` object of the process.
        """
        logger.info(f"Reading folder {folder_path}")
        paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        data = []
        for path in paths:
            logger.info(f"=> Processing {path}")

            # If the path is a folder, recursively read the folder.
            if os.path.isdir(path):
                response = self.__read_table_folder(path, creator, accept_duplicates)
                logger.info(response.message)
                data.extend(response.data["tables"])
                continue

            response = self.__read_table_file(path, creator, accept_duplicates)
            logger.info(
                f"==> Processing table {path} {response.status.value}: {response.message}"
            )
            data.append(response.data)

        file_count = len(data)
        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"{file_count} files in folder {folder_path} has been processed.",
            data={"file_count": file_count, "tables": data},
        )

    def __read_metadata_file(
        self, metadata_path: str, table_id: str
    ) -> Response:
        """
        Reads a metadata file (CSV or TXT) and registers it in the database.

        ## Args
        - **metadata_path** (`str`): The path to a specific metadata file (`CSV`
        or `TXT`).
        - **table_id** (`str`): The associated table of the metadata.

        ## Returns
        - `Response`: A `Response` object of the process.
        """
        # Index -1 to get the file extension, then slice [1:] to remove the dot.
        file_type = os.path.splitext(metadata_path)[-1][1:]

        if file_type not in ["txt", "csv"]:
            return Response(
                status=ResponseStatus.ERROR,
                message="Invalid file type. Please use 'txt' or 'csv'.",
            )

        if file_type == "txt":
            with open(metadata_path, "r") as f:
                metadata_content = f.read()

            return self.__insert_metadata(metadata_content, table_id)

        if file_type == "csv":
            metadata_df = pd.read_csv(metadata_path)
            data = []
            for index, row in metadata_df.iterrows():
                table_id = row["table_id"]
                metadata_content = row["value"]
                response = self.__insert_metadata(
                    metadata_content, table_id
                )
                logger.info(response.message)
                data.extend(response.data["metadata_ids"])

            return Response(
                status=ResponseStatus.SUCCESS,
                message=f"{len(metadata_df)} metadata entries has been added.",
                data={"file_count": len(data), "metadata_ids": data},
            )

        return None

    def __read_metadata_folder(
        self, metadata_path: str, table_id: str
    ) -> Response:
        """
        Reads a folder and registers all of its metadata to the database.

        ## Args
        - **metadata_path** (`str`): The path to a folder containing metadata files.
        - **table_id** (`str`): The associated table of the metadata files.

        ## Returns
        - `Response`: A `Response` object of the process.
        """
        logger.info(f"Reading metadata folder {metadata_path}")
        paths = [os.path.join(metadata_path, f) for f in os.listdir(metadata_path)]
        metadata_ids = []
        for path in paths:
            logger.info(f"=> Processing {path}")

            # If the path is a folder, recursively read the folder.
            if os.path.isdir(path):
                response = self.__read_metadata_folder(path, table_id)
                logger.info(response.message)
                metadata_ids.extend(response.data["metadata_ids"])
                continue

            response = self.__read_metadata_file(path, table_id)
            logger.info(
                f"==> Processing metadata {path} {response.status.value}: {response.message}"
            )
            metadata_ids.extend(response.data["metadata_ids"])

        file_count = len(metadata_ids)
        return Response(
            status=ResponseStatus.SUCCESS,
            message=f"{file_count} files in folder {metadata_path} has been processed.",
            data={"file_count": file_count, "metadata_ids": metadata_ids},
        )

    def __insert_metadata(
        self, metadata_content: str, table_id: str
    ) -> Response:
        """
        Inserts metadata associated with table `table_id` into the database.

        ## Args
        - **metadata_content** (`str`): The content of the metadata.
        - **table_id** (`str`): The associated table of the metadata.

        ## Returns
        - `Response`: A `Response` object of the process.
        """
        try:
            with duckdb.connect(self.db_path) as connection:
                payload = {
                    "payload": metadata_content.strip(),
                }

                metadata_id = connection.sql(
                    f"""INSERT INTO table_contexts (table_id, context)
                    VALUES ('{table_id}', '{json.dumps(payload)}')
                    RETURNING id"""
                ).fetchone()[0]

                return Response(
                    status=ResponseStatus.SUCCESS,
                    message=f"Metadata created with ID: {metadata_id}",
                    data={"file_count": 1, "metadata_ids": [metadata_id]},
                )
        except Exception as e:
            return Response(
                status=ResponseStatus.ERROR,
                message=f"Error connecting to database: {e}",
            )


if __name__ == "__main__":
    fire.Fire(Registrar)
