"""Silver to Gold — balance, label, stratify, and export ML-ready dataset."""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, rand, row_number, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window


def main() -> None:
    spark = SparkSession.builder \
        .appName("PaintingInAPainting-SilverToGold") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # load silver data
    df = spark.read.csv("data/silver", header=True, inferSchema=True)
    print(f"Silver rows: {df.count()}")

    # cap each style at 3000 images to reduce class imbalance
    MAX_PER_STYLE = 3000
    window = Window.partitionBy("style").orderBy(rand(seed=42))
    df = df.withColumn("row_num", row_number().over(window))
    df_balanced = df.filter(col("row_num") <= MAX_PER_STYLE).drop("row_num")
    print(f"After balancing: {df_balanced.count()}")

    # style label mapping (0 to N-1)
    styles = df_balanced.select("style").distinct().orderBy("style").collect()
    style_mapping = {row["style"]: idx for idx, row in enumerate(styles)}
    print(f"Style classes: {len(style_mapping)}")

    # artist label mapping (only artists with 10+ paintings)
    artist_counts = df_balanced.groupBy("artist").agg(count("*").alias("cnt"))
    valid_artists = artist_counts.filter(col("cnt") >= 10).select("artist").orderBy("artist").collect()
    artist_mapping = {row["artist"]: idx for idx, row in enumerate(valid_artists)}
    print(f"Artist classes: {len(artist_mapping)} (filtered to 10+ paintings)")

    # genre label mapping
    genres = df_balanced.select("primary_genre").distinct().orderBy("primary_genre").collect()
    genre_mapping = {row["primary_genre"]: idx for idx, row in enumerate(genres)}
    print(f"Genre classes: {len(genre_mapping)}")

    # add integer label columns
    style_udf = udf(lambda x: style_mapping.get(x, -1), IntegerType())
    artist_udf = udf(lambda x: artist_mapping.get(x, -1), IntegerType())
    genre_udf = udf(lambda x: genre_mapping.get(x, -1), IntegerType())

    df_labeled = df_balanced \
        .withColumn("style_idx", style_udf(col("style"))) \
        .withColumn("artist_idx", artist_udf(col("artist"))) \
        .withColumn("genre_idx", genre_udf(col("primary_genre")))

    print(f"Artists with mapping: {df_labeled.filter(col('artist_idx') >= 0).count()}")
    print(f"Artists without mapping: {df_labeled.filter(col('artist_idx') == -1).count()}")

    # stratified split: 80% train, 10% val, 10% test
    train_fractions = {style: 0.8 for style in style_mapping}
    df_train = df_labeled.sampleBy("style", fractions=train_fractions, seed=42)

    df_remaining = df_labeled.subtract(df_train)
    val_fractions = {style: 0.5 for style in style_mapping}
    df_val = df_remaining.sampleBy("style", fractions=val_fractions, seed=42)

    df_test = df_remaining.subtract(df_val)

    df_train = df_train.withColumn("split", lit("train"))
    df_val = df_val.withColumn("split", lit("val"))
    df_test = df_test.withColumn("split", lit("test"))

    df_gold = df_train.union(df_val).union(df_test).drop("subset")

    print(f"Train: {df_train.count()}")
    print(f"Val: {df_val.count()}")
    print(f"Test: {df_test.count()}")
    print(f"Total: {df_gold.count()}")

    # save gold dataset and mappings as CSVs
    out = Path("data/gold/labels")
    out.mkdir(parents=True, exist_ok=True)

    df_gold.toPandas().to_csv(out / "gold_wikiart.csv", index=False)

    style_rows = [(k, v) for k, v in style_mapping.items()]
    genre_rows = [(k, v) for k, v in genre_mapping.items()]
    artist_rows = [(k, v) for k, v in artist_mapping.items()]

    spark.createDataFrame(style_rows, ["style", "style_idx"]) \
        .toPandas().to_csv(out / "gold_style_mapping.csv", index=False)
    spark.createDataFrame(artist_rows, ["artist", "artist_idx"]) \
        .toPandas().to_csv(out / "gold_artist_mapping.csv", index=False)
    spark.createDataFrame(genre_rows, ["genre", "genre_idx"]) \
        .toPandas().to_csv(out / "gold_genre_mapping.csv", index=False)

    print(f"\nGold data saved to {out}")

    # final verification
    print(f"\nUnique styles: {df_gold.select('style').distinct().count()}")
    print(f"Unique artists: {df_gold.select('artist').distinct().count()}")
    print(f"Unique genres: {df_gold.select('primary_genre').distinct().count()}")
    df_gold.groupBy("split").agg(count("*").alias("count")).orderBy("split").show()

    spark.stop()


if __name__ == "__main__":
    main()