"""Bronze to Silver - read Bronze Parquet from ADLS, clean, persist Silver Parquet back.

Silver is the cleaning layer. It reads the validated Bronze data and applies:
- Duplicate removal (phash)
- Invalid row removal (uncertain artists)
- Genre string cleaning
- Artist name standardization
- Dimension filtering

Pipeline format: Raw (CSV) -> Bronze (Parquet) -> Silver (Parquet) -> Gold (Parquet + CSV)
"""

import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, regexp_replace, split, trim

from src.utils.datalake import DataLakeClient


def find_part_file(spark_output_dir: Path, extension: str) -> Path:
    """Find the single part file in a Spark output directory."""
    for f in spark_output_dir.iterdir():
        if f.name.startswith("part-") and f.name.endswith(extension):
            return f
    raise FileNotFoundError(f"No {extension} part file found in {spark_output_dir}")


def main() -> None:
    lake = DataLakeClient()

    # download bronze parquet from ADLS
    local_bronze = Path("data/bronze/bronze_wikiart.parquet")
    print("Downloading Bronze from ADLS...")
    lake.download_file("bronze/wikiart/bronze_wikiart.parquet", local_bronze)

    spark = SparkSession.builder \
        .appName("PaintingInAPainting-BronzeToSilver") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # read bronze parquet
    df = spark.read.parquet(str(local_bronze))
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

    # persist silver as parquet
    silver_dir = Path("data/silver")
    spark_tmp = silver_dir / "_spark_output"
    df_silver.coalesce(1).write.mode("overwrite").parquet(str(spark_tmp))

    part_file = find_part_file(spark_tmp, ".parquet")
    clean_path = silver_dir / "silver_wikiart.parquet"
    shutil.copy2(str(part_file), str(clean_path))
    shutil.rmtree(str(spark_tmp))

    print(f"Silver saved locally. Rows: {df_silver.count()}")
    print(f"  Format: Parquet")
    print(f"Unique styles: {df_silver.select('style').distinct().count()}")
    print(f"Unique artists: {df_silver.select('artist').distinct().count()}")
    print(f"Unique genres: {df_silver.select('primary_genre').distinct().count()}")

    spark.stop()

    # upload silver to ADLS
    print("Uploading Silver to ADLS...")
    lake.upload_file(clean_path, "silver/wikiart/silver_wikiart.parquet")
    print("Done.")


if __name__ == "__main__":
    main()