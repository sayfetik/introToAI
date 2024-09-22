count(10) :- write(10), nl.  
count(N) :- 
    N < 10,          
    write(N), nl,    
    N1 is N + 1,     
    count(N1).       

start_counting :- count(1).
