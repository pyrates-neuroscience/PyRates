# Specification of the YAML-based template specification language

PyRates employs a domain specific language based on the [YAML](https://yaml.org/) 
standard (version 1.2) to define templates. Templates are building blocks of a network
that can be reused across multiple scales. The following is a summary which fields are
allowed or needed to be defined on a single template. 

## Operator template

```yaml
OperatorTemplateName:  # this is the operator name
  description: "..." # (optional) description text (goes into the object's __doc__)
  base: OperatorTemplate  # reference to the Python object that is used as basis
  label: "..." # (optional) alternative name for display
  equations: # single equation string or list of equation strings
  variables: # further information to define aspects of variables
    var_name:  # must be the same as used in the equation
      description: "..." # (optional) explanation
      unit: "..." # (optional) unit of the variable (only as additional info, not used) 
      default:  # defines data type, variable type and/or default value
        # can be any of the following:
        # - `float` or `int` with default value in brackets, e.g. `float(1.2)`
        # - plain default value, e.g. `0.37`
        # - `input` or `output`
        # if a value is given, the variable will be treated as a constant, without as 
        # state variable
```

## Node template

```yaml
NodeTemplateName:  # the name of the node template
  description: "..." # (optional) description text (goes into the object's __doc__)
  base: NodeTemplate  # reference to the Python object that is used as basis
  label: "..." # (optional) alternative name for display
  operators:  # list operators or key-value pairs with operator as key and changes as values
    - OperatorA
    - OperatorB
```

## Edge template

Edge templates are structured just like node templates but with `EdgeTemplate` as base

## Circuit template

```yaml
CircuitTemplateName:  # the name of the circuit template
  description: "..." # (optional) description text (goes into the object's __doc__)
  base: CircuitTemplate
  label: "..." # (optional) alternative name for display
  nodes: # key-value pairs with internal names of nodes as keys and their template as value
    MyNode1: NodeTemplateName
    MyNode2: NodeTemplateName
  circuits: # key-value pairs with internal names of sub-circuits as keys and their template as value
    SubCircuit1: OtherCircuitTemplateName
    SubCircuit2: OtherCircuitTemplateName
  edges: # list of edges
    # [source, target, type, {weight, delay, other variables}]
    - [MyNode1/OperatorA/var_name, MyNode2/OperatorA/var_name, EdgeTemplateName, {weight: 1, delay: 0.1, op/var_name: 3.1}]
    - [SubCircuit1/node/op/var, MyNode1/op/var, EdgeTempalteName, {weight: 3, ...}]
```

## Template inheritance

Template can inherit from other templates. To do this, the parent template needs to be specified as `base`. All other +
information given in the new template will be considered as changes to the parent template. For example:
```yaml
MyOperator:
    base: OperatorTemplateName
    variables:
      var_name:
        default: 2
``` 

## Adapting templates when called

When a template is referenced in another template, it can also be changed on-the-fly. 

```yaml
MyNode:
  operators:
    MyOperator:
      variables:
        var_name: 3  # this way the default will be overwritten (but you could also explicitly say "default")
```

## Template paths and YAML aliases

When referencing a template inside the same file, the template name is sufficient. For referencing templates outside the
current file, it is necessary to also mention the path to that template. Paths can be either given as
absolute or relative paths with slashes `/`, e.g. `../path/to/template`. If the template can be found in a Python module
inside the current environment, the syntax is equivalent to Python imports: `path.to.template`.
In both cases above, the last element is interpreted as template name and the one before as filename of the YAML file. 
The parser recognizes the file extensions `.yml` and `.yaml`.

To avoid repeating long template paths, the aliasing syntax built into YAML can be used:
```yaml
aliases:  # this keyword is not necessary, but improves readability
  - &ShortName very/long/template/path/that/should/not/be/typed/too/many/times
  - &Shorty2 another/long/template/name/that/is/really/annoying

MyCircuit:
  ...
  nodes: 
    A: *ShortName
    B: *ShortName
  edges:
    - [A/Op/Var, B/Op/Var, *Shorty2, {weight: 10}]
``` 
