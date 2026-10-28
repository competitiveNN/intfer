# mini-clone README -- CITATION FIXTURE, not a real upstream README.

This tree exists so `scripts/matrices/mutate_engine_citations.sh` has an
`upstream/` tree to resolve against that is IN-REPO and travels with the
checkout.  Every citation in the fixture corpus is hosted here rather than at a
real upstream clone, so the matrix is operable on a box with no
`/var/home/fra/dev/intfer-upstream/` at all -- a matrix that cannot run is a
matrix that certifies nothing.

It carries enough lines that a fixture citation can name a real line.
