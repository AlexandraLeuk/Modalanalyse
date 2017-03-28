Second Meeting on 28.3.2017

Team **Modal Monsters**

# Eigenvalue Problem and Git

Present is
+ Michael Zauner
+ Jan Valasek 
+ Sebastian Thormann
+ Alexandra Leukauf

| Function      	  | Person           |
| :--- |---|
| Chairmen      | Sebastian Thormann |
| Secretary     | Michael Zauner       |


## Main discussion
We discussed each eigenvalue solver, which are mentioned in homework and we also decided to share our work by using GitHub.  Michael explained his program for the first 4 eigenvalue solvers (vector iteration,2 inverse vector iteration, Rayleigh quotient iteration and higher eigenvalues using inverse vector iteration in orthogonal direction). Afterwards we tested the eigenvalue solver programs and spoke about the occurring phenomenons.

## Distribution of work

| Person | Work package |
| --- | --- |
| Michael | done |
| Jan | Subspace-iteration |
| Sebastian | Eigenvalues and mode shapes |
| Alexandra | QR-algorithm |

## Main difficulty 
We instructed how to use sourcetree. Downloading it and how to use Git-repos


Inverse Vector Iteration: When we chose a shift point of 1.2, the program´s result for the eigenvalue was 1.4, even though 1.4 shouldn´t be an eigenvalue.
Main insight:
Rayleigh: The inverse matrix becomes singular, if sigma is chosen as an eigenvalue. The program´s eigenvectors 
Eigenvalues converge faster than their eigenvectors
If 2 eigenvalues are close to each other (e.g. 3.9 and 4), the Rayleigh quotient iteration converges slower to the bigger value.
 
**Next meeting** 28.3.2017 12:00 BA 1.Stock -  preparing for the workshop
