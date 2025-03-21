How does the model decide to feature to select and the sp?

IG ( f, sp ) =  I(parent) - ( Nleft / N * I(left) + Nright / N * I(right)  )

f: feature 
sp: split-point

impurity criteria (Controlled by criterion parameter while initialized the model)
* Gini index (DEfault option, faster)
* entropy

They usually lead to similar results

If we work we an unconstrained Tree nodes are grown recursively.  At each node it splits data to maximize the above formula. If IG(node) = 0 declare the node a leaf
