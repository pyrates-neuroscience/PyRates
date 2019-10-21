# Changelog

## 0.8

### 0.8.0 (work in progress)

- removed version ID numbers of operator/node instances in the intermediate representation. I.e. a node label `mynode` 
  was previously renamed to `mynode.0` and will now keep it's original label.
- moved all functionality of ComputeGraph into CircuitIR, which is now the main interface for the backend. 
  - `CircuitIR` now has a `.compile` method that performs all vectorization and transformation into the computable 
    backend form.
- vectorization will transform all nodes into instances of `VectorizedNodeIR` that have labels like `vector_nodeX` with 
  X being a integer index. The map between old nodes and vectorized nodes with respective index is saved in the 
  `label_map` dictionary attribute of the `CircuitIR`
- When adding input or sampling output of a network with multiple stacked levels of circuits, you can now use `all` to 
  get all nodes within that particular level. For example `mysubcircuit1/all/mynode` will get all nodes with label 
  `mynode` that are in one level of sub-circuits below `mysubcircuit`.
- Tensorflow support now relies on the current 2.0 release candidate `tensorflow-2.0-rc`
- Added optional install requirements via `extras_require` in setup.py 