from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, trim, regexp_replace

def iniciar_spark():
    return SparkSession.builder \
        .appName("ENDES_ETL_Limpieza_Fix") \
        .getOrCreate()

def limpiar_nombres_cols(df):
    new_cols = [c.strip().upper() for c in df.columns]
    return df.toDF(*new_cols)

def limpiar_id_agresivo(df, col_name):
    """
    Elimina TODOS los espacios en blanco dentro del ID.
    Ejemplo: "001  2" -> "0012"
    Esto asegura que el cruce de tablas no falle por espacios invisibles.
    """
    return df.withColumn(col_name, regexp_replace(col(col_name), "\\s+", ""))

def ejecutar_etl():
    spark = iniciar_spark()
    spark.sparkContext.setLogLevel("ERROR")
    
    print(">>> [ETL] Cargando CSVs crudos...")
    # Cargar CSVs
    try:
        rec41 = spark.read.option("header", "true").option("delimiter", ";").option("inferSchema", "true").csv("data/raw/REC41.csv")
        rec94 = spark.read.option("header", "true").option("delimiter", ";").option("inferSchema", "true").csv("data/raw/REC94.csv")
        re516 = spark.read.option("header", "true").option("delimiter", ";").option("inferSchema", "true").csv("data/raw/RE516171.csv")
        rec84dv = spark.read.option("header", "true").option("delimiter", ";").option("inferSchema", "true").csv("data/raw/REC84DV_2024.csv")
    except Exception as e:
        print(f"Error leyendo archivos: {e}")
        return

    # 1. Estandarizar nombres de columnas
    rec41 = limpiar_nombres_cols(rec41)
    rec94 = limpiar_nombres_cols(rec94)
    re516 = limpiar_nombres_cols(re516)
    rec84dv = limpiar_nombres_cols(rec84dv)

    # 2. LIMPIEZA AGRESIVA DE IDs (EL SECRETO)
    print(">>> [ETL] Normalizando IDs (eliminando espacios)...")
    rec41 = limpiar_id_agresivo(rec41, "CASEID")
    rec94 = limpiar_id_agresivo(rec94, "CASEID")
    re516 = limpiar_id_agresivo(re516, "CASEID")
    rec84dv = limpiar_id_agresivo(rec84dv, "CASEID")

    # ---------------------------------------------------------
    # PROCESO 1: EMBARAZO
    # ---------------------------------------------------------
    print(">>> [ETL] Cruzando tablas Embarazo...")
    if "IDX94" in rec94.columns:
        rec94 = rec94.withColumnRenamed("IDX94", "MIDX")

    df_risk = rec41.join(rec94, on=['CASEID', 'MIDX'], how='left') \
                   .join(re516, on='CASEID', how='left')

    # Target Riesgo
    df_risk = df_risk.withColumn("TARGET_RIESGO", 
        when((col("M14") < 4) | (col("M15").isin(11, 12, 96)), 1)
        .otherwise(0)
    )

    cols_risk = ['CASEID', 'MIDX', 'TARGET_RIESGO', 'V012', 'V501', 'V714', 'V701', 'M13', 'M14', 'M15', 'V190']
    valid_cols = [c for c in cols_risk if c in df_risk.columns]
    df_master_risk = df_risk.select(*valid_cols).na.fill(0)
    
    print(f"   -> Registros Embarazo generados: {df_master_risk.count()}")
    df_master_risk.coalesce(1).write.mode('overwrite').option("header", "true").csv("data/processed/master_embarazo.csv")

    # ---------------------------------------------------------
    # PROCESO 2: VIOLENCIA
    # ---------------------------------------------------------
    print(">>> [ETL] Cruzando tablas Violencia...")
    
    # Inner Join ahora debería funcionar mucho mejor
    df_viol = rec84dv.join(re516, on='CASEID', how='inner')

    # Detectar violencia
    viol_check_cols = [c for c in df_viol.columns if c.startswith('D105') or c.startswith('D101')]
    condition = lit(0)
    for c in viol_check_cols:
        condition = condition + when(col(c).cast("int").isin(1, 2), 1).otherwise(0)
    
    df_viol = df_viol.withColumn("TARGET_VIOLENCIA", when(condition > 0, 1).otherwise(0))

    cols_viol = ['CASEID', 'TARGET_VIOLENCIA', 'V501', 'V701', 'V714', 'V743A', 'D113', 'V025', 'V190']
    valid_cols_v = [c for c in cols_viol if c in df_viol.columns]
    df_master_viol = df_viol.select(*valid_cols_v).na.fill(0)

    print(f"   -> Registros Violencia generados: {df_master_viol.count()}")
    df_master_viol.coalesce(1).write.mode('overwrite').option("header", "true").csv("data/processed/master_violencia.csv")
    
    print(">>> ETL Finalizado.")
    spark.stop()

if __name__ == "__main__":
    ejecutar_etl()