% name, surname, age, height (sm), weight (kg)
person(timur, gatin, 60, 175, 80).
person(aidar, gatin, 35, 180, 85).
person(iskander, gatin, 10, 140, 40).
person(farid, hamedov, 40, 178, 82).
person(emil, hamedov, 15, 160, 50).
person(alsu, gatin, 58, 165, 65).
person(aigul, gatin, 33, 170, 68).
person(leyla, faritova, 34, 168, 60).

male(timur).
male(aidar).
male(iskander).
male(farid).
male(emil).

female(alsu).
female(aigul).
female(leyla).

parent(timur, aidar).
parent(timur, aigul).
parent(alsu, aidar).
parent(alsu, aigul).

parent(aidar, iskander).
parent(leyla, iskander).

parent(aigul, emil).
parent(farid, emil).

mother(X, Y) :- parent(X, Y), female(X).
father(X, Y) :- parent(X, Y), male(X).
sister(X, Y) :- parent(Z, X), parent(Z, Y), female(X), X \= Y.
brother(X, Y) :- parent(Z, X), parent(Z, Y), male(X), X \= Y.
grandfather(X, Y) :- parent(X, Z), parent(Z, Y), male(X).
grandmother(X, Y) :- parent(X, Z), parent(Z, Y), female(X).
aunt(X, Y) :- sister(X, Z), parent(Z, Y).
uncle(X, Y) :- brother(X, Z), parent(Z, Y).

people_with_surname(Surname) :- person(Name, Surname, _, _, _), write(Name), nl, fail.
people_with_surname(_).

people_with_age(Age) :- person(Name, _, Age, _, _), write(Name), nl, fail.
people_with_age(_).

people_with_height(Height) :- person(Name, _, _, Height, _), write(Name), nl, fail.
people_with_height(_).

people_with_weight(Weight) :- person(Name, _, _, _, Weight), write(Name), nl, fail.
people_with_weight(_).
