# De Tweede Kamer Analyse Tool
Deze applicatie biedt een interactieve analyse- en zoekomgeving voor openbare documenten van de Tweede Kamer der Staten-Generaal.
De tool is bedoeld om grote hoeveelheden Kamerstukken overzichtelijk te maken en eenvoudige analyses mogelijk te maken op basis van inhoud, tijd en documenttype.
De applicatie draait op Streamlit Community Cloud en gebruikt uitsluitend openbare data.

# 🔎 Zoeken in Kamerstukken
Zoeken op één of meerdere zoektermen
AND / OR-logica
Zoekt in alle relevante velden van een document

# 📊 Analyse & visualisatie
Aantal documenten per maand
Trendanalyse met lineaire regressie
Verdeling van documenttypes (bijv. Kamervragen, brieven, verslagen)

# 📄 Directe toegang tot bron
Klikbare links naar originele documenten op de officiële website van de Tweede Kamer

# 🕒 Dagelijks geactualiseerde dataset
Data wordt automatisch elke dag ververst via GitHub Actions
De Streamlit-app zelf doet géén API-calls bij het opstarten

# Deze applicatie maakt gebruik van de Tweede Kamer Open Data API: 
https://gegevensmagazijn.tweedekamer.nl/

Datumselectie vanaf eind 2019

De data wordt opgeslagen in Parquet-formaat voor snelle laadtijden en efficiënte verwerking.

# Transparantie & disclaimer
Deze applicatie gebruikt uitsluitend openbare overheidsdata.
De analyses en visualisaties zijn bedoeld ter ondersteuning van inzicht en onderzoek.
Interpretaties of conclusies uit deze tool vertegenwoordigen geen officiële standpunten van de Tweede Kamer of de Rijksoverheid.
De ontwikkelaar is niet verantwoordelijk voor de inhoud van de brondata.

# Licentie
Dit project is vrij te gebruiken voor educatieve, journalistieke en onderzoeksdoeleinden.
Voor commercieel gebruik: neem contact op met mij aub (antwoord is toch nee).

# Contact
Voor vragen, feedback of suggesties:
LinkedIn: https://www.linkedin.com/in/jkpstuve/
