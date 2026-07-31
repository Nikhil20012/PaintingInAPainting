"""Bronze to Silver — download bronze from ADLS, clean, persist, upload back."""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, split, trim

from src.utils.datalake import DataLakeClient


def main() -> None:
    lake = DataLakeClient()

    # download bronze from ADLS
    local_bronze = Path("data/bronze/bronze_wikiart.csv")
    print("Downloading Bronze from ADLS...")
    lake.download_file("bronze/wikiart/bronze_wikiart.csv", local_bronze)

    spark = SparkSession.builder \
        .appName("PaintingInAPainting-BronzeToSilver") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # load bronze data
    df = spark.read.csv(str(local_bronze), header=True, inferSchema=True)
    print(f"Bronze rows: {df.count()}")

    # extract style from filename
    df = df.withColumn("style", split(col("filename"), "/")[0])

    # remove uncertain artist rows
    df = df.filter(col("subset") != "uncertain artist")
    print(f"After removing uncertain artists: {df.count()}")

    # remove phash duplicates
    df = df.dropDuplicates(["phash"])
    print(f"After removing duplicates: {df.count()}")

    # clean genre column
    df = df.withColumn(
        "primary_genre",
        regexp_replace(
            split(
                regexp_replace(col("genre"), "[\\[\\]']", ""),
                ","
            )[0],
            "^ | $", ""
        )
    )

    # trim and standardize artist names
    df = df.withColumn("artist", trim(col("artist")))
    df = df.withColumn("artist", regexp_replace(col("artist"), "\\s+", " "))

    # filter extreme dimensions
    df = df.filter(
        (col("width") >= 200) &
        (col("height") >= 200) &
        (col("width") <= 10000) &
        (col("height") <= 10000)
    )
    print(f"After dimension filter: {df.count()}")

    # select and reorder final columns
    df_silver = df.select(
        "filename",
        "style",
        "artist",
        "primary_genre",
        "description",
        "phash",
        "width",
        "height",
        "genre_count",
        "subset",
    )

    # persist silver locally
    silver_dir = Path("data/silver")
    silver_dir.mkdir(parents=True, exist_ok=True)
    df_silver.toPandas().to_csv(silver_dir / "silver_wikiart.csv", index=False)
    print(f"Silver saved locally. Rows: {df_silver.count()}")

    # final check
    print(f"Unique styles: {df_silver.select('style').distinct().count()}")
    print(f"Unique artists: {df_silver.select('artist').distinct().count()}")
    print(f"Unique genres: {df_silver.select('primary_genre').distinct().count()}")

    spark.stop()

    # upload silver to ADLS
    print("Uploading Silver to ADLS...")
    lake.upload_file(silver_dir / "silver_wikiart.csv", "silver/wikiart/silver_wikiart.csv")
    print("Done.")


if __name__ == "__main__":
    main()