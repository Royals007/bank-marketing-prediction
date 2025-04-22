# Bankmarketing-Vorhersage – Usecase Studie

Weitere Informationen zum Projekt findest du in der Datei task.md.

So führst du das Projekt aus
Umgebung einrichten:

    conda env create -f env.yml

Führe die Datei model_pred.py aus, um folgende Schritte zu starten:

- Vorverarbeitung der Daten

- Training des Modells

- Evalution der Modellleistung

Speichern von Diagrammen und Berichten in /images und /reports

## Einzelne Module ausführen
    python src/data_preprocess.py  
    python src/model_training.py  
    python src/model_pred.py

## Modellleistung
Das Modell wird mit den folgenden Metriken bewertet:

- Genauigkeit (Accuracy)

- Präzision (Precision)

- Recall

- F1-Score

- ROC AUC-Score (inkl. Diagramm)

- Konfusionsmatrix (inkl. Diagramm)

Ein vollständiger Bericht wird exportiert nach:

    /images/classification_report.csv

## Visualisierungen
Leistungsvisualisierungen (gespeichert in /images/):

- Konfusionsmatrix

- ROC-Kurve

# Referenzen
Datensatz: UCI Bank Marketing Dataset (siehe am Ordner data)

Sprache: Python 3.12+

ML-Bibliotheken: pandas, scikit-learn, matplotlib, seaborn, numpy, jupyter notebook