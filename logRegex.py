import re

"""
Estrazione dell'indirizzo IP: Trova tutti gli indirizzi IP.
Estrazione del timestamp: Trova la data e l'ora in cui è stata fatta la richiesta.
Estrazione del metodo HTTP e URL: Trova il metodo HTTP (GET, POST, ecc.) e l'URL richiesto.
Estrazione del codice di stato HTTP: Trova il codice di stato HTTP.
Estrazione della dimensione della risposta: Trova la dimensione della risposta.
"""

with open('logFile.txt', 'r') as file:
    log = file.read()

#partiamo dall'assunzione che siano corretti
ip_pt=r"((\d{1,3}\.){3}\d{1,3})"
ips=re.finditer(ip_pt,log)
for match in ips:
    print(match.group())

