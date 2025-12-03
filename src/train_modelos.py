import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col, udf, lit, explode, array
from pyspark.sql.types import DoubleType, FloatType
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Configuración Docker
matplotlib.use('Agg')

def iniciar_spark():
    spark = SparkSession.builder \
        .appName("ENDES_ML_Final_Full") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR") 
    return spark

def preparar_datos_robustos(df, target_col, features_cols):
    # 1. Convertir Target
    df_clean = df.withColumn(target_col, col(target_col).cast(DoubleType()))
    # 2. Convertir Features
    for c in features_cols:
        df_clean = df_clean.withColumn(c, col(c).cast(DoubleType()))
    # 3. Rellenar Nulos con 0
    return df_clean.na.fill(0.0)

def balancear_clases_spark(df, target_col):
    """Función de Oversampling para corregir desbalance"""
    count_neg = df.filter(col(target_col) == 0.0).count()
    count_pos = df.filter(col(target_col) == 1.0).count()
    
    if count_pos == 0 or count_neg == 0: return df
        
    if count_neg > count_pos:
        ratio = int(count_neg / count_pos)
        major_df = df.filter(col(target_col) == 0.0)
        minor_df = df.filter(col(target_col) == 1.0)
        
        # Multiplicar minoría
        oversampled_minor = minor_df.withColumn("dummy", explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
        print(f"   [BALANCEO] Clase 0: {count_neg} | Clase 1: {count_pos} -> {oversampled_minor.count()}")
        return major_df.unionAll(oversampled_minor)
    return df

def guardar_graficos(predictions_df, nombre_algoritmo, nombre_dataset, target_col):
    try:
        data_pd = predictions_df.select(
            col(target_col).alias("label"), 
            col("prediction"), 
            col("probability")
        ).toPandas()

        y_true = data_pd["label"]
        y_pred = data_pd["prediction"]
        
        # REPORTE DETALLADO EN CONSOLA
        print(f"\n   >>> REPORTE: {nombre_algoritmo}")
        print(classification_report(y_true, y_pred, target_names=['Negativo', 'Positivo']))
        print("   " + "-"*50)

        # Probabilidad para ROC
        def get_prob(v):
            try: return float(v[1])
            except: return 0.0
        y_prob = data_pd["probability"].apply(get_prob)

        clean_name = nombre_algoritmo.strip().replace(" ", "_")
        clean_dataset = nombre_dataset.strip().replace(" ", "_")
        
        # 1. Matriz Confusión
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusión - {nombre_algoritmo}\n({nombre_dataset})')
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        plt.savefig(f"outputs/figures/{clean_dataset}_{clean_name}_CM.png")
        plt.close()

        # 2. ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
        plt.title(f'ROC - {nombre_algoritmo}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"outputs/figures/{clean_dataset}_{clean_name}_ROC.png")
        plt.close()
        
    except Exception as e:
        print(f"   [WARNING] Gráfico falló: {e}")

def evaluar_comparativa(df, target_col, feature_cols, nombre_dataset):
    print(f"\n" + "="*60)
    print(f" ANALIZANDO Y GUARDANDO: {nombre_dataset}".center(60))
    print("="*60)
    
    df = preparar_datos_robustos(df, target_col, feature_cols)
    if df.count() == 0: return []

    train_data_raw, test_data = df.randomSplit([0.7, 0.3], seed=42)
    train_data = balancear_clases_spark(train_data_raw, target_col)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    
    algoritmos = [
        ("GradientBoosting", GBTClassifier(labelCol=target_col, featuresCol="features", maxIter=20, seed=42)),
        ("RandomForest", RandomForestClassifier(labelCol=target_col, featuresCol="features", numTrees=50, seed=42))
    ]
    
    mejor_auc = 0
    mejor_modelo = None
    mejor_nombre = ""

    for nombre, algoritmo in algoritmos:
        try:
            pipeline = Pipeline(stages=[assembler, algoritmo])
            model = pipeline.fit(train_data)
            predictions = model.transform(test_data)
            
            eval_auc = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC").evaluate(predictions)
            print(f"Modelo: {nombre} | AUC: {eval_auc:.4f}")

            # Generar Gráficos y Reportes
            guardar_graficos(predictions, nombre, nombre_dataset, target_col)

            if eval_auc > mejor_auc:
                mejor_auc = eval_auc
                mejor_modelo = model
                mejor_nombre = nombre

        except Exception as e:
            print(f"Error en {nombre}: {e}")

    # GUARDAR EL MODELO GANADOR
    if mejor_modelo:
        ruta_modelo = f"data/processed/modelo_{nombre_dataset}"
        print(f">>> Guardando el mejor modelo ({mejor_nombre}) en: {ruta_modelo}")
        mejor_modelo.write().overwrite().save(ruta_modelo)

    return []

def ejecutar_modeling():
    spark = iniciar_spark()
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    
    print(">>> Cargando Datos...")
    try:
        df_emb = spark.read.option("header", "true").csv("data/processed/master_embarazo.csv")
        df_vio = spark.read.option("header", "true").csv("data/processed/master_violencia.csv")
    except: return

    cols_no_emb = ['CASEID', 'MIDX', 'TARGET_RIESGO']
    pred_emb = [c for c in df_emb.columns if c not in cols_no_emb]
    
    cols_no_vio = ['CASEID', 'TARGET_VIOLENCIA']
    pred_vio = [c for c in df_vio.columns if c not in cols_no_vio]

    evaluar_comparativa(df_emb, 'TARGET_RIESGO', pred_emb, "RIESGO_EMBARAZO")
    evaluar_comparativa(df_vio, 'TARGET_VIOLENCIA', pred_vio, "VIOLENCIA_FAMILIAR")
    
    spark.stop()

if __name__ == "__main__":
    ejecutar_modeling()