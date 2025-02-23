import os

from pneuma.registrar.registrar import Registrar

DB_PATH = "../out/storage.db"


def read_benchmark_data(registration: Registrar):
    registration.read_folder("../../data_src/tables/chicago_open_data", "sample_user")


def read_sample_data(registration: Registrar):
    registration.read_table("../sample_data/csv/5cq6-qygt.csv", "sample_user")
    registration.read_table("../sample_data/csv/5n77-2d6a.csv", "sample_user")

    registration.add_context(
        "../sample_data/csv/5cq6-qygt.csv", "../sample_data/context/sample_context.txt"
    )
    registration.add_context(
        "../sample_data/csv/5n77-2d6a.csv", "../sample_data/context/sample_context.txt"
    )

    registration.add_summary(
        "../sample_data/csv/5cq6-qygt.csv", "../sample_data/summary/sample_summary.txt"
    )
    registration.add_summary(
        "../sample_data/csv/5n77-2d6a.csv", "../sample_data/summary/sample_summary.txt"
    )


if __name__ == "__main__":
    os.makedirs("../out", exist_ok=True)
    registration = Registrar(DB_PATH)
    registration.setup()

    read_sample_data(registration)
