# Tests

## Presolve

* [x] Rectangular hess
* [x] Wrong c dims
* [ ] Non positive semi-definite hess
* [x] inf/nan values

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
* [x] inf/nan values

### Bounds

* [x] Base, no ineq, 2 lb, 2 ub, 1 redundant, 1 extra
* [x] +∞ lower bound
* [x] -∞ upper bound
* [x] Upper bound \< lower bound
* [x] Ignore (None, None), (-∞, +∞), (None, +∞), (-∞, None) bounds
* [x] Ignore bounds beyond # vars
* [x] inf/nan values

### Inequality constraints

* [x] Base full problem read
* [x] Wrong b, a dims
  * [x] Wrong # of cols
  * [x] # of rows different
  * [x] b, no a
  * [x] a, no b
* [x] a has zero rows
* [x] a has zero rows and negative rhs
* [x] a has singleton rows
  * [x] new lower bound
  * [x] new upper bound
  * [x] inconsistent lower bound
  * [x] inconsistent upper bound
  * [x] redundant bounds
* [x] inf/nan values
