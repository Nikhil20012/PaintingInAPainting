"""Bronze layer - validate, audit, and persist raw WikiArt metadata.

Bronze is the ingestion layer. It does NOT clean, filter, or remove data.
It validates the schema, attaches ingestion metadata, reports data quality
issues, and persists the validated dataset as Parquet. Cleaning belongs in Silver.

Why each addition belongs in Bronze (not Silver):
- Schema validation: catches source format changes at ingestion time,
  before any downstream processing runs on broken data.
- Ingestion metadata: traces when data was ingested, from which source,
  under which schema version. This is lineage, not transformation.
- Style extraction from filename: this is parsing the source format,
  not a business transformation. The style is encoded in the file path
  by WikiArt, not derived by our pipeline.
- Data quality report: Bronze reports what's wrong. Silver fixes it.
  This separation means Silver's cleaning rules can change without
  re-ingesting the data.

Pipeline format: Raw (CSV) -> Bronze (Parquet) -> Silver (Parquet) -> Gold (Parquet + CSV)
Parquet provides columnar storage, compression, predicate pushdown, faster Spark
scans, and schema preservation across the pipeline.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    avg, col, count, countDistinct, lit, max, min, round, split,
    sum as spark_sum, when,
)
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from src.utils.datalake import DataLakeClient


# expected schema for WikiArt metadata CSV
EXPECTED_SCHEMA = StructType([
    StructField("filename",    StringType(),  nullable=False),
    StructField("artist",      StringType(),  nullable=False),
    StructField("genre",       StringType(),  nullable=True),
    StructField("description", StringType(),  nullable=True),
    StructField("phash",       StringType(),  nullable=False),
    StructField("width",       IntegerType(), nullable=False),
    StructField("height",      IntegerType(), nullable=False),
    StructField("genre_count", IntegerType(), nullable=True),
    StructField("subset",      StringType(),  nullable=False),
])

REQUIRED_COLUMNS = ["filename", "artist", "phash", "width", "height", "subset"]
SCHEMA_VERSION = "1.0"
SOURCE_DATASET = "wikiart-kaggle"


def validate_schema(df: DataFrame) -> dict:
    """Validate that the DataFrame matches the expected schema.

    Returns a validation report dict. Does NOT modify the data.
    """
    report = {
        "schema_valid": True,
        "missing_columns": [],
        "type_mismatches": [],
        "unexpected_columns": [],
    }

    actual_cols = {f.name: f.dataType for f in df.schema.fields}
    expected_cols = {f.name: f.dataType for f in EXPECTED_SCHEMA.fields}

    for col_name in expected_cols:
        if col_name not in actual_cols:
            report["missing_columns"].append(col_name)
            report["schema_valid"] = False

    for col_name, expected_type in expected_cols.items():
        if col_name in actual_cols and actual_cols[col_name] != expected_type:
            report["type_mismatches"].append({
                "column": col_name,
                "expected": str(expected_type),
                "actual": str(actual_cols[col_name]),
            })
            report["schema_valid"] = False

    for col_name in actual_cols:
        if col_name not in expected_cols:
            report["unexpected_columns"].append(col_name)

    return report


def validate_required_columns(df: DataFrame) -> dict:
    """Check that required columns have no nulls.

    Reports null counts for required fields only. Does NOT drop rows.
    """
    null_exprs = [
        spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in REQUIRED_COLUMNS
    ]
    null_row = df.select(null_exprs).collect()[0]
    return {c: null_row[c] for c in REQUIRED_COLUMNS if null_row[c] > 0}


def compute_null_counts(df: DataFrame) -> dict:
    """Count nulls across all columns in a single pass."""
    null_exprs = [
        spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    null_row = df.select(null_exprs).collect()[0]
    return {c: null_row[c] for c in df.columns if null_row[c] > 0}


def compute_quality_report(df: DataFrame) -> dict:
    """Produce a data quality report for audit purposes.

    Reports issues without modifying the data. Silver uses this
    information to decide what to clean.
    """
    total = df.count()
    null_counts = compute_null_counts(df)

    unique_phash = df.select("phash").distinct().count()
    duplicate_phash = total - unique_phash

    dim_stats = df.select(
        min("width").alias("min_width"),
        max("width").alias("max_width"),
        round(avg("width"), 0).alias("avg_width"),
        min("height").alias("min_height"),
        max("height").alias("max_height"),
        round(avg("height"), 0).alias("avg_height"),
    ).collect()[0]

    style_counts = df.groupBy("styles") \
        .agg(count("*").alias("count")) \
        .orderBy(col("count").desc()) \
        .collect()

    unique_artists = df.select(countDistinct("artist")).collect()[0][0]
    multi_genre = df.filter(col("genre_count") > 1).count()
    unique_genres = df.select("genre").distinct().count()

    subset_counts = df.groupBy("subset") \
        .agg(count("*").alias("count")) \
        .collect()

    return {
        "record_count": total,
        "null_counts": null_counts,
        "duplicate_phash_count": duplicate_phash,
        "unique_phash_count": unique_phash,
        "dimensions": {
            "min_width": int(dim_stats["min_width"]),
            "max_width": int(dim_stats["max_width"]),
            "avg_width": int(dim_stats["avg_width"]),
            "min_height": int(dim_stats["min_height"]),
            "max_height": int(dim_stats["max_height"]),
            "avg_height": int(dim_stats["avg_height"]),
        },
        "unique_artists": unique_artists,
        "unique_styles": len(style_counts),
        "style_distribution": {row["styles"]: row["count"] for row in style_counts},
        "multi_genre_entries": multi_genre,
        "unique_genre_strings": unique_genres,
        "subset_distribution": {row["subset"]: row["count"] for row in subset_counts},
    }


def add_ingestion_metadata(df: DataFrame, ingestion_id: str) -> DataFrame:
    """Attach ingestion metadata columns.

    These columns trace lineage: when was this data ingested, from
    which source, under which schema version. This is metadata about
    the ingestion event, not a transformation of the data itself.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    return df \
        .withColumn("ingestion_timestamp", lit(timestamp)) \
        .withColumn("source_dataset", lit(SOURCE_DATASET)) \
        .withColumn("schema_version", lit(SCHEMA_VERSION)) \
        .withColumn("ingestion_id", lit(ingestion_id))


def find_part_file(spark_output_dir: Path, extension: str) -> Path:
    """Find the single part file in a Spark output directory."""
    for f in spark_output_dir.iterdir():
        if f.name.startswith("part-") and f.name.endswith(extension):
            return f
    raise FileNotFoundError(f"No {extension} part file found in {spark_output_dir}")


def main() -> None:
    lake = DataLakeClient()
    ingestion_id = str(uuid4())[:8]

    # download raw metadata from ADLS
    local_raw = Path("data/raw/classes.csv")
    print("Downloading raw metadata from ADLS...")
    lake.download_file("raw/wikiart/classes.csv", local_raw)

    spark = SparkSession.builder \
        .appName("PaintingInAPainting-BronzeIngest") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # load raw metadata with enforced schema
    df = spark.read.schema(EXPECTED_SCHEMA).option("header", True).csv(str(local_raw))

    # schema validation
    print("\nSchema validation...")
    schema_report = validate_schema(df)
    if schema_report["schema_valid"]:
        print("  Schema: VALID")
    else:
        print("  Schema: INVALID")
        if schema_report["missing_columns"]:
            print(f"  Missing columns: {schema_report['missing_columns']}")
        if schema_report["type_mismatches"]:
            for m in schema_report["type_mismatches"]:
                print(f"  Type mismatch: {m['column']} (expected {m['expected']}, got {m['actual']})")
    if schema_report["unexpected_columns"]:
        print(f"  Unexpected columns: {schema_report['unexpected_columns']}")

    # required column null validation
    print("\nRequired column validation...")
    required_violations = validate_required_columns(df)
    if required_violations:
        print(f"  Null violations in required columns: {required_violations}")
    else:
        print("  All required columns are non-null: PASS")

    # extract style from filename (parsing source format, not a business transformation)
    df = df.withColumn("styles", split(col("filename"), "/")[0])

    # data quality report
    print("\nData quality report...")
    quality_report = compute_quality_report(df)
    print(f"  Record count: {quality_report['record_count']:,}")
    print(f"  Null counts: {quality_report['null_counts'] or 'none'}")
    print(f"  Duplicate phash: {quality_report['duplicate_phash_count']}")
    print(f"  Dimensions: {quality_report['dimensions']}")
    print(f"  Unique styles: {quality_report['unique_styles']}")
    print(f"  Unique artists: {quality_report['unique_artists']}")
    print(f"  Multi-genre entries: {quality_report['multi_genre_entries']}")
    print(f"  Unique genre strings: {quality_report['unique_genre_strings']}")
    print(f"  Subset distribution: {quality_report['subset_distribution']}")

    # add ingestion metadata
    df = add_ingestion_metadata(df, ingestion_id)

    # persist bronze layer as Parquet
    bronze_dir = Path("data/bronze")
    spark_tmp = bronze_dir / "_spark_output"
    df.coalesce(1).write.mode("overwrite").parquet(str(spark_tmp))

    # move the part file to a clean path
    part_file = find_part_file(spark_tmp, ".parquet")
    clean_path = bronze_dir / "bronze_wikiart.parquet"
    shutil.copy2(str(part_file), str(clean_path))
    shutil.rmtree(str(spark_tmp))

    print(f"\nBronze saved locally to {bronze_dir}")
    print(f"  Format: Parquet")
    print(f"  Ingestion ID: {ingestion_id}")
    print(f"  Schema version: {SCHEMA_VERSION}")
    print(f"  Source: {SOURCE_DATASET}")

    # save reports
    reports = {
        "ingestion_id": ingestion_id,
        "record_count": quality_report["record_count"],
        "schema_version": SCHEMA_VERSION,
        "source_dataset": SOURCE_DATASET,
        "schema_validation": schema_report,
        "required_column_violations": required_violations,
        "data_quality": quality_report,
    }
    with open(bronze_dir / "bronze_report.json", "w") as f:
        json.dump(reports, f, indent=2)
    print(f"  Quality report saved to {bronze_dir / 'bronze_report.json'}")

    spark.stop()

    # upload bronze to ADLS
    print("\nUploading Bronze to ADLS...")
    lake.upload_file(clean_path, "bronze/wikiart/bronze_wikiart.parquet")
    lake.upload_file(bronze_dir / "bronze_report.json", "bronze/wikiart/bronze_report.json")
    print("Done.")


if __name__ == "__main__":
    main()