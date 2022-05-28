## BEFORE

$ perf stat ./target/release/difft sample_files/elisp_before.el sample_files/elisp_after.el > /dev/null

 Performance counter stats for './target/release/difft sample_files/elisp_before.el sample_files/elisp_after.el':

             11.43 msec task-clock:u              #    0.958 CPUs utilized          
                 0      context-switches:u        #    0.000 /sec                   
                 0      cpu-migrations:u          #    0.000 /sec                   
               346      page-faults:u             #   30.284 K/sec                  
        18,743,529      cycles:u                  #    1.641 GHz                    
        39,259,117      instructions:u            #    2.09  insn per cycle         
         7,818,915      branches:u                #  684.355 M/sec                  
           121,697      branch-misses:u           #    1.56% of all branches        

       0.011925419 seconds time elapsed

       0.008910000 seconds user
       0.002949000 seconds sys


~/projects/difftastic 
$ perf stat ./target/release/difft sample_files/load_before.js sample_files/load_after.js > /dev/null

 Performance counter stats for './target/release/difft sample_files/load_before.js sample_files/load_after.js':

            127.70 msec task-clock:u              #    0.943 CPUs utilized          
                 0      context-switches:u        #    0.000 /sec                   
                 0      cpu-migrations:u          #    0.000 /sec                   
               815      page-faults:u             #    6.382 K/sec                  
       233,672,757      cycles:u                  #    1.830 GHz                    
       424,039,682      instructions:u            #    1.81  insn per cycle         
        77,019,087      branches:u                #  603.147 M/sec                  
         1,369,605      branch-misses:u           #    1.78% of all branches        

       0.135397567 seconds time elapsed

       0.113261000 seconds user
       0.014927000 seconds sys


~/projects/difftastic 
$ perf stat ./target/release/difft sample_files/Session_before.kt sample_files/Session_after.kt > /dev/null

 Performance counter stats for './target/release/difft sample_files/Session_before.kt sample_files/Session_after.kt':

            543.66 msec task-clock:u              #    0.991 CPUs utilized          
                 0      context-switches:u        #    0.000 /sec                   
                 0      cpu-migrations:u          #    0.000 /sec                   
             2,565      page-faults:u             #    4.718 K/sec                  
     1,509,659,051      cycles:u                  #    2.777 GHz                    
     4,333,385,467      instructions:u            #    2.87  insn per cycle         
     1,064,060,361      branches:u                #    1.957 G/sec                  
         4,196,728      branch-misses:u           #    0.39% of all branches        

       0.548373686 seconds time elapsed

       0.535438000 seconds user
       0.007834000 seconds sys

## AFTER

$ perf stat ./target/release/difft sample_files/elisp_before.el sample_files/elisp_after.el > /dev/null

 Performance counter stats for './target/release/difft sample_files/elisp_before.el sample_files/elisp_after.el':

              8.23 msec task-clock:u              #    0.951 CPUs utilized          
                 0      context-switches:u        #    0.000 /sec                   
                 0      cpu-migrations:u          #    0.000 /sec                   
               347      page-faults:u             #   42.163 K/sec                  
        17,674,848      cycles:u                  #    2.148 GHz                    
        39,174,417      instructions:u            #    2.22  insn per cycle         
         7,815,908      branches:u                #  949.680 M/sec                  
           119,791      branch-misses:u           #    1.53% of all branches        

       0.008655226 seconds time elapsed

       0.004330000 seconds user
       0.004339000 seconds sys


~/projects/difftastic 
$ perf stat ./target/release/difft sample_files/load_before.js sample_files/load_after.js > /dev/null

 Performance counter stats for './target/release/difft sample_files/load_before.js sample_files/load_after.js':

             87.02 msec task-clock:u              #    0.991 CPUs utilized          
                 0      context-switches:u        #    0.000 /sec                   
                 0      cpu-migrations:u          #    0.000 /sec                   
               815      page-faults:u             #    9.366 K/sec                  
       229,182,109      cycles:u                  #    2.634 GHz                    
       423,819,533      instructions:u            #    1.85  insn per cycle         
        77,012,250      branches:u                #  885.033 M/sec                  
         1,318,617      branch-misses:u           #    1.71% of all branches        

       0.087846126 seconds time elapsed

       0.079116000 seconds user
       0.007936000 seconds sys


~/projects/difftastic 
$ perf stat ./target/release/difft sample_files/Session_before.kt sample_files/Session_after.kt > /dev/null

 Performance counter stats for './target/release/difft sample_files/Session_before.kt sample_files/Session_after.kt':

            661.72 msec task-clock:u              #    0.998 CPUs utilized          
                 0      context-switches:u        #    0.000 /sec                   
                 0      cpu-migrations:u          #    0.000 /sec                   
             2,059      page-faults:u             #    3.112 K/sec                  
     1,467,515,601      cycles:u                  #    2.218 GHz                    
     4,333,238,540      instructions:u            #    2.95  insn per cycle         
     1,064,055,442      branches:u                #    1.608 G/sec                  
         3,522,118      branch-misses:u           #    0.33% of all branches        

       0.663331220 seconds time elapsed

       0.656490000 seconds user
       0.003962000 seconds sys
