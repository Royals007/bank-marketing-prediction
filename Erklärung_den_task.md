# Das bedeutet im Detail:

1.Entwickeln Sie ein Arbeitsprodukt (z.B. Jupyter Notebook) welches aufzeigt, wie Sie in der explorativen Datenanalyse vorgehen.

Lösungen: 

In notebooks/eda.ipynb wird die eda-Analyse erläutert. Hier hat Herr Nanduru die angegebenen Rohdaten aus den Marketingdaten der UCI Bank entnommen (angegebener Link von UCI). 

- Im Datenordner gibt es 4 Arten von csv-Dateien. Nach Prüfung der Ergebnisse der einzelnen Datensätze. Die Datei additional-bank-full.csv wird aufgrund der Anzahl der Attribute (21) und der Größe der Daten ausgewählt.

    - Für die künftige Modellauswahl und Vorhersageanalyse wird dieser Datensatz hilfreicher sein als die anderen Datensätze.

- Für die EDA-Analyse untersucht Herr.Nanduru die Daten vor und nach jedem Bauteil und identifiziert die Merkmale des Bauteils. Diese werden in einem Jupyter-Notebook (.ipynb-Datei) dargestellt.

    - Fehlende Werte

    - Doppelte Informationen

    - Ausreißer (Outliers)

    - Korrelationsmatrix

    - kategoriale Umsetzungen

- Schließlich wurden die „Rohdaten“ verarbeitet, bereinigt und als „processed_data" gespeichert. Diese Daten werden für die Aufgabe des maschinellen Lernens zur Vorhersage verwendet.

2.Definieren Sie eine/mehrere geeignete Zielmetrik/en, um die Performance einer Prognose messen zu können. Begründen Sie ebenfalls, wieso Sie sich für ihre Zielmetrik entschieden haben.

Lösungen :

Verschiedene Zielmetriken auf der Grundlage des gegebenen Datensatzes und der Vorhersageaufgabe berücksichtigt. Nämlich,

- Accuracy

    Basierend auf dem erhaltenen Datensatz sind die Klassen für die Zielvariable unzureichend, da die meisten Kunden nicht eingeloggt sind (die Mehrheit ist „nein“).

- Recall

    Sie stellt sicher, dass die meisten Kunden sich tatsächlich angemeldet haben („ja“).

- Precision / Präzision

    Liefert die richtigen Informationen, die nicht fälschlicherweise Kunden ansprechen, die kein Abonnement abschließen wollen.

- F1-Score

    Gute allgemeine metrische Informationen.

- ROC AUC

    Liefert eine schwellenunabhängige Ansicht der Leistung

- confusion matrix /Konfusionsmatrix

    Es handelt sich um eine binäre Klassifizierungsaufgabe. Sie liefert die True Positive (TP - richtig vorhergesagt „ja“), True Negative (TN - richtig vorhergesagt „nein“), False Positives (FP - vorhergesagt „ja“, aber „nein“), False Negative (FN - vorhergesagt „nein“, aber „ja“).

    Auch nützlich, um die Metriken zu berechnen und die Fehler des Modells zu identifizieren.


3.Implementieren und trainieren Sie ein Modell, um die Zielvariable «y» vorauszusagen.
    - Nutzen Sie hier die Programmiersprache und Libraries mit welchen Sie am meisten vertraut sind.

Lösungen: 

siehe die Python-Dateien (Projekt)

4.Veranschaulichen Sie die Erkenntnisse und Performance der Prognosen kurz in geeigneter Form.

Lösungen: 

Mit Random-Forest-Algorithmus, um die Zielvariable "y" mit 100 Estimators zu finden und random_state ist 42 und Key Performance der verschiedenen Metriken und Visualisierungen sind in der gespeichert,

    .images/example_metrics_images.png

und die Metrikinformationen werden im csv-Format gespeichert, d. h,

    images/classification_matrix

Aus dem Bericht geht hervor, dass

    Accuracy - 90.80%  (ML-Modell macht gute Vorhersagen)
    Klassen-Implanz für
        - no: F1_score 0.95% (hervorragende Leistung)
        - yes: F1_score 0.53% (Erfassen von TP)

Kann sich auf das Geschäft auswirken, wenn das Ziel darin besteht, alle potenziellen Abonnenten zu identifizieren.


# Weiterführende Fragen:
5. Würden Sie für die Voraussage noch weitere Datenpunkte in Erwägung ziehen? 	
    - Wenn ja: welche Informationen (die eine Bank typischerweise zur Verfügung hat) wären am vielversprechendsten?

Lösungen: 

Offensichtlich führen mehr Daten zu besseren Ergebnissen bei der ML-Vorhersage. Wenn die Bank vielversprechende Informationen liefern kann, wie 

- SHUFA-Infos

- Kundenreaktionsgeschichte

- Nutzung von Bankprodukten

- wie oft digitale Nutzung, wenn sie Zugang haben, sonst nicht

- Marktzinsen saisonal oder monatlich

- Nützliche Vorteile für Kunden

Mit diesen Informationen werden die Datenmerkmale erweitert und die Identifizierung der Zielvariablen und ihrer abhängigen Faktoren führt zu einer wesentlich höheren Präzision und Genauigkeit bei der Modellschulung.


6.Können die Erkenntnisse Ihrer Analyse auch für andere Bereiche der Bank verwendet werden? 
    - Muss dafür die Problemstellung angepasst werden? Wie würden Sie hier vorgehen?

Lösungen: 

Natürlich deshalb, weil der Datensatz mit mehr Attributen betrachtet wird, so dass das Modell gut trainieren kann. Diese Analyse kann auf andere Bereiche des Bankennetzwerks ausgedehnt werden, wie z. B.

- Fraud detection
- SHUFA (Kredit, Darlehen, etc)
- Kundenaktivitäten (mit Mitarbeitern, Konto geschlossen)

Lösungen den neu Problemstellung:

- Neudefinition des Ziels auf der Grundlage der Anforderungen

- Sicherstellen, dass die Historie der Stichprobendaten aufgezeichnet wird

- Anpassung der neuen Merkmale

- Überprüfung der Konsistenz der Daten

- Trainieren des Modells mit einem bestimmten Ziel unter Berücksichtigung verschiedener Komponenten, Metriken und Optimierer usw.

- Beobachten Sie das Ergebnis und integrieren Sie es in zukünftige Banksysteme, die auf dem Ziel basieren. Wenn es beispielsweise auf digitalen/Marketingbereichen basiert, können Sie es in Marketingbereiche und CRM usw. integrieren.

Schließlich wird das gesamte ML-Training, der Einsatz und die Überwachung variiert und je nach Datensatz und Zielvariable verwendet, um durch die Implementierung verschiedener Algorithmen und anderer Komponenten bessere Ergebnisse zu erzielen.




Ich danke Ihnen für Ihre Zeit und Geduld. Bitte lass es mich wissen, falls Sie weitere Informationen oder Erklärungen benötigen.

Ich freue mich auf den nächsten Meilenstein auf unserer Reise.

MFG

