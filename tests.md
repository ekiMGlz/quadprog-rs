# Tests

## Presolve

* [x] Rectangular hess
* [x] Wrong c dims
* [ ] Non positive semi-definite hess

### Eq constraints

* [x] base problem with a zero row
* [x] \# non zero eq constraints \> \# of vars
* [x] Wrong b_eq, a_eq dims
  * [x] Wrong # of cols
  * [x] # of rows different
  * [x] b_eq, no a_eq
  * [x] a_eq, no b_eq
* [x] a_eq has zero rows with non-zero rhs
* [x] a_eq is rank deficient
* [x] a_eq has zero rows and is rank deficient

### Bounds

* [x] Base, no ineq, 2 lb, 2 ub, 1 redundant, 1 extra
* [ ] +∞ lower bound
* [ ] -∞ upper bound
* [ ] Upper bound \< lower bound
* [ ] Ignore (None, None), (-∞, +∞), (None, +∞), (-∞, None) bounds
* [ ] Ignore bounds beyond # vars

### Inequality constraints

* [ ] Base full problem read
* [ ] Wrong b, a dims
  * [ ] Wrong # of cols
  * [ ] # of rows different
  * [ ] b, no a
  * [ ] a, no b
* [ ] a has zero rows
* [ ] a has zero rows and negative rhs
* [ ] a has singleton rows
  * [ ] new lower bound
  * [ ] new upper bound
  * [ ] inconsistent bound
  * [ ] redundant bound
* [ ] correct ineq matrix and rhs construction
  * [ ] No lb
  * [ ] No ub
  * [ ] No ineq
  * [ ] Only ineq
  * [ ] Only lb
  * [ ] Only ub
