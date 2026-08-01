"""Silver to Gold - read Silver Parquet from ADLS, balance, label, split, persist back.

Gold is the ML-ready layer. It reads cleaned Silver data and applies:
- Class balancing (cap styles at 3000)
- Integer label encoding (style, artist, genre)
- Stratified train/val/test split (80/10/10)

Output: Parquet for pipeline consistency + CSV for downstream ML code compatibility.
Pipeline format: Raw (CSV) -> Bronze (Parquet) -> Silver (Parquet) -> Gold (Parquet + CSV)
"""

import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, rand, row_number, udf
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

from src.utils.datalake import DataLakeClient


def find_part_file(spark_output_dir: Path, extension: str) -> Path:
    """Find the single part file in a Spark output directory."""
    for f in spark_output_dir.iterdir():
        if f.name.startswith("part-") and f.name.endswith(extension):
            return f
    raise FileNotFoundError(f"No {extension} part file found in {spark_output_dir}")


def main() -> None:
    lake = DataLakeClient()

    # download silver parquet from ADLS
    local_silver = Path("data/silver/silver_wikiart.parquet")
    print("Downloading Silver from ADLS...")
    lake.download_file("silver/wikiart/silver_wikiart.parquet", local_silver)

    spark = SparkSession.builder \
        .appName("PaintingInAPainting-SilverToGold") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # read silver parquet
    df = spark.read.parquet(str(local_silver))
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

    # persist gold as parquet (pipeline consistency)
    gold_dir = Path("data/gold")
    spark_tmp = gold_dir / "_spark_output"
    df_gold.coalesce(1).write.mode("overwrite").parquet(str(spark_tmp))

    parquet_dir = gold_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    part_file = find_part_file(spark_tmp, ".parquet")
    gold_parquet = parquet_dir / "gold_wikiart.parquet"
    shutil.copy2(str(part_file), str(gold_parquet))
    shutil.rmtree(str(spark_tmp))

    # persist gold as CSV (downstream ML code compatibility)
    csv_dir = gold_dir / "labels"
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_gold.toPandas().to_csv(csv_dir / "gold_wikiart.csv", index=False)

    # save mapping CSVs (small reference files, CSV is appropriate)
    style_rows = [(k, v) for k, v in style_mapping.items()]
    genre_rows = [(k, v) for k, v in genre_mapping.items()]
    artist_rows = [(k, v) for k, v in artist_mapping.items()]

    spark.createDataFrame(style_rows, ["style", "style_idx"]) \
        .toPandas().to_csv(csv_dir / "gold_style_mapping.csv", index=False)
    spark.createDataFrame(artist_rows, ["artist", "artist_idx"]) \
        .toPandas().to_csv(csv_dir / "gold_artist_mapping.csv", index=False)
    spark.createDataFrame(genre_rows, ["genre", "genre_idx"]) \
        .toPandas().to_csv(csv_dir / "gold_genre_mapping.csv", index=False)

    print(f"\nGold saved locally")
    print(f"  Parquet: {gold_parquet}")
    print(f"  CSVs: {csv_dir}")

    # final verification
    print(f"Unique styles: {df_gold.select('style').distinct().count()}")
    print(f"Unique artists: {df_gold.select('artist').distinct().count()}")
    print(f"Unique genres: {df_gold.select('primary_genre').distinct().count()}")
    df_gold.groupBy("split").agg(count("*").alias("count")).orderBy("split").show()

    spark.stop()

    # upload gold to ADLS
    print("Uploading Gold to ADLS...")
    lake.upload_file(gold_parquet, "gold/wikiart/gold_wikiart.parquet")
    lake.upload_file(csv_dir / "gold_wikiart.csv", "gold/wikiart/labels/gold_wikiart.csv")
    lake.upload_file(csv_dir / "gold_style_mapping.csv", "gold/wikiart/labels/gold_style_mapping.csv")
    lake.upload_file(csv_dir / "gold_artist_mapping.csv", "gold/wikiart/labels/gold_artist_mapping.csv")
    lake.upload_file(csv_dir / "gold_genre_mapping.csv", "gold/wikiart/labels/gold_genre_mapping.csv")
    print("Done.")


if __name__ == "__main__":
    main()