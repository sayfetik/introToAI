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
