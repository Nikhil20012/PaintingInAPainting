"""Bronze layer audit — profile the raw WikiArt metadata."""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, col, count, countDistinct, max, min, round, split,
)


def main() -> None:
    spark = SparkSession.builder \
        .appName("PaintingInAPainting-BronzeIngest") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # load raw metadata CSV
    df = spark.read.csv(
        "data/wikiart/classes.csv",
        header=True,
        inferSchema=True,
    )

    total = df.count()
    print(f"\nTotal rows: {total:,}")
    df.printSchema()
    df.show(5, truncate=False)

    # extract style from filename (folder before '/')
    df = df.withColumn("styles", split(col("filename"), "/")[0])

    # style distribution
    style_count = df.select("styles").distinct().count()
    print(f"\nDistinct styles: {style_count}")
    df.groupBy("styles") \
        .agg(count("*").alias("image_count")) \
        .orderBy(col("image_count").desc()) \
        .show(30, truncate=False)

    # artist distribution
    unique_artists = df.select(countDistinct("artist")).collect()[0][0]
    print(f"Unique artists: {unique_artists}")
    df.groupBy("artist") \
        .agg(count("*").alias("painting_count")) \
        .orderBy(col("painting_count").desc()) \
        .show(10, truncate=False)

    # genre analysis
    multi_genre = df.filter(col("genre_count") > 1).count()
    unique_genres = df.select("genre").distinct().count()
    print(f"Multi-genre entries: {multi_genre}")
    print(f"Unique genre strings: {unique_genres}")

    # image dimensions
    df.select(
        min("width").alias("min_width"),
        max("width").alias("max_width"),
        round(avg("width"), 0).alias("avg_width"),
        min("height").alias("min_height"),
        max("height").alias("max_height"),
        round(avg("height"), 0).alias("avg_height"),
    ).show()

    # duplicate detection via perceptual hash
    unique_phash = df.select("phash").distinct().count()
    duplicates = total - unique_phash
    print(f"Phash duplicates: {duplicates}")

    # train/test split distribution
    df.groupBy("subset") \
        .agg(count("*").alias("count")) \
        .orderBy("subset") \
        .show()

    # summary
    print(f"Total rows: {total:,}")
    print(f"Distinct styles: {style_count}")
    print(f"Unique artists: {unique_artists}")
    print(f"Multi-genre entries: {multi_genre}")
    print(f"Unique genre strings: {unique_genres}")
    print(f"Phash duplicates: {duplicates}")

    spark.stop()


if __name__ == "__main__":
    main()