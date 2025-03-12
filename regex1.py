import re

txt="Ciao! Mi chiamo Marco Rossi e lavoro come sviluppatore web. Il mio indirizzo email è marco.rossi@example.com, mentre il mio collega utilizza luca.bianchi@azienda.it. " \
"Se hai bisogno di contattarmi, chiamami al +39 345 6789123 oppure al fisso 06-9876543. " \
"Abbiamo una riunione importante il 12/03/2024, ma la data potrebbe cambiare al 15/04/2024. " \
"Ricorda di controllare il sito dell'azienda: https://www.azienda.it/eventi. Se il link non funziona, prova a visitare http://backup.azienda.it.\n" \
"Nel frattempo, ho trovato alcuni errori nel report:\n" \
"->errore nel calcolo delle vendite\n" \
"->errore nei dati di febbraio\n" \
"Dobbiamo correggerli al più presto!\n" \
"Ah, dimenticavo! La mia nuova password è Pa55word! (anche se dovrei sceglierne una più sicura)."

"""
1->Trova tutte le parole di 5 lettere
2->Cerca i numeri di telefono
3->Estrai tutti i link presenti
4->Sostituisci "errore" con "correzione"
5->Cambia il formato delle date
6->Verifica se le email sono valide
7->Controlla se la password è abbastanza sicura
"""

pt1=r"\b\w{5}\b"
match1=re.findall(pt1,txt)
print(match1)

pt2=r"(\+\d\d\s?\d\d\d\s?\d{7}|\d\d-\d{7})"
match2=re.findall(pt2,txt)
print(match2)

sub=re.sub("errore","correzione",txt)
print(sub)

pattern = r"(\d{2})/(\d{2})/(\d{4})"  
replacement = r"\1-\2-\3"

sub_txt = re.sub(pattern, replacement, txt)
print(sub_txt)

email_check=r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
emails=re.findall(email_check,txt)
print(emails)

link=r"https?://([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})(/[a-zA-Z0-9\-]*)*"
links=re.findall(link,txt)
print(links)

pass_validation=r"^(?=[A-Z]\w{8,}[!$%&?@#]+)"


while not re.match(pass_validation,txt):
    print("Password non sicura, prova a migliorarla.")
    txt=input()

print("Password sicura")

