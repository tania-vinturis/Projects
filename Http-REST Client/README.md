# Pasii de rezolvare:

1. Initializare si citire comenzi:
    - Am folosit bucla WHILE pt a rula continuu pana cand se introduce de a tastatura "exit"
    - Am citit comenzile cu ajutorul functiei "scanf", si am evaluat fiecare comanda folosind "strcmp" pt a determina ce trebuie sa fac mai departe.
2. Functia "performUserOperation":
    - Este apelata pt comenzi legate de user: "register", "login", "logout"
    - Pt "register" si "login", se citeste numele de utilizator, cat si parola, se creeaza un obiect JSON cu aceste date si se trimite o cerere HTTP POST la server.
    - Raspunsul serverului este tratat de functii specifice:
        - "extractCookieAndHandleResponse" pt "login"
        - "handleStandardResponse" pt "register" si "logout"
3. Functia "performLibraryOperation":
    - Este apelata pt comenzi legate de library, cu ar fi adaugarea sau stergerea cartilor
    - Construieste URL-ul adecvat in functie de operatia specifica si trimite cereri: GET, POST, DELETE la server, folosind cooki-uri si tokenul obtinut in urma comenzii "entry_library"
4. Manipulare JSON:
    - am integrat fisierele parson.c, parson.h(din linkul GitHub din enuntul temei)
    - am folosit acest fisier .c pt a crea si parsa obiecte JSON
    - datele pt fiecare carte sunt citite de la tastatura si introduse in obiectul JSON
5. Tratare raspunsuri server:
    - Dupa fiecare cerere HTTP, clietul primeste un raspuns de la server pe care il proceseaza pt a extrage informtii precum: cookie-uri, token-uri, mesaje de eroare/succes
6. Conexiune la server:
    - am folosit functii din laboratorul 9 (helpers.c si majoritatea fisierului request.c)

