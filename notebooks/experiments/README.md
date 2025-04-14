## Instructions for reproducing experiments

### Mallows simulation (Figure 1)

Run the following command to reproduce Figure 1:
```
python3 fig1.py --n 20 --m 40 --curves 10 --curve_pts 50 --ff <plot destination file>
```

Parameters:
- `n`: number of papers in each sample
- `m`: number of authors in each sample
- `curves`: number of sample curves to generate
- `curve_pts`: number of gammas to examine
- `ff`: place to save plot image

### Experiments

#### Figure 2a

Run the following command to reproduce Figure 2a:

```
python3 fig2a.py --n 20 --m 40 --curves 10 --curve_pts 50 --clusters 25 --components 2 --df <data source file> --ff <plot destination file>
```

Parameters:
- `n`: number of papers in each sample
- `m`: number of authors in each sample
- `curves`: number of sample curves to generate
- `curve_pts`: number of gammas to examine
- `clusters`: number of k-Means clusters to generate author sub-groups
- `components`: number of PCA components to preprocess author embeddings for k-Means clustering
- `df`: location of data source file:
    - data should be a dictionary with two entries, "authors" and "papers"
    - "authors": list of dictionaries each representing an author's data. Each dictionary must have a field "embedding", which is a list of the embeddings of the author's papers.
    - "papers": list of paper embeddings
- `ff`: place to save plot image

#### Figure 2b

Run the following command to reproduce Figure 2b:
```
python3 fig2b.py --n 20 --m 40 --curves 10 --curve_pts 50 --beta 0.9 --df <data source file> --ff <plot destination file>
```

Parameters:
- `n`: number of papers in each sample
- `m`: number of authors in each sample
- `curves`: number of sample curves to generate
- `curve_pts`: number of gammas to examine
- `beta`: proportion of the population to be correctly estimated
- `df`: location of data source file:
    - data should be a dictionary with two entries, "authors" and "papers"
    - "authors": list of dictionaries each representing an author's data. Each dictionary must have a field "embedding", which is a matrix of the embeddings of the author's papers (papers x embedding dimension).
    - "papers": matrix of paper embeddings (papers x embedding dimension)
- `ff`: place to save plot image
