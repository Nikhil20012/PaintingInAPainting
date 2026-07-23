"""Bronze to Silver — clean, deduplicate, and standardize WikiArt metadata."""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, split, trim


def main() -> None:
    spark = SparkSession.builder \
        .appName("PaintingInAPainting-BronzeToSilver") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # load bronze data
    df = spark.read.csv(
        "data/wikiart/classes.csv",
        header=True,
        inferSchema=True,
    )
    print(f"Bronze rows: {df.count()}")

    # extract style from filename
    df = df.withColumn("style", split(col("filename"), "/")[0])

    # remove uncertain artist rows
    df = df.filter(col("subset") != "uncertain artist")
    print(f"After removing uncertain artists: {df.count()}")

    # remove phash duplicates
    df = df.dropDuplicates(["phash"])
    print(f"After removing duplicates: {df.count()}")

    # clean genre column — extract primary genre from messy string format
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

    # save as silver CSV
    df_silver.coalesce(1).write.mode("overwrite").option("header", True).csv("data/silver")
    print(f"Silver saved. Rows: {df_silver.count()}")

    # final check
    print(f"Unique styles: {df_silver.select('style').distinct().count()}")
    print(f"Unique artists: {df_silver.select('artist').distinct().count()}")
    print(f"Unique genres: {df_silver.select('primary_genre').distinct().count()}")
    df_silver.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()